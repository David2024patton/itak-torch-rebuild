# weights.mojo - DeviceWeights: the Glimmer weights resident
# on device, one buffer per tensor slot (Wave C lane C1; the
# seam contract is WAVE_C_INTERFACES.md).
#
# The CPU GlimmerModel stays the loader of record: its cached
# bytes are the upload source, uploaded ONCE here at build.
# Quantized and F16 slots become uint8 device buffers over the
# slot's raw bytes; F32 slots become float32 device buffers
# over the slot's values. The final-norm values (the model's
# final_norm list, the output_norm.weight tensor, which the
# CPU model holds outside its slot list) get their own float32
# buffer.
#
# The exact resident byte total is computed from the loader's
# own figures BEFORE any buffer is created
# (weights_total_bytes + GpuRuntime.reserve_weights, which
# refuses gpu_budget_exceeded by name), then buffers are
# created; refuse first, allocate second.
#
# Accessor shape: a meta accessor returns the slot's
# DeviceSlotMeta (name, type_id, rows, cols, resident bytes,
# storage kind); the matching buffer accessor (slot_u8 /
# slot_f32 and the named embedding / output forms) returns a
# view carrying the buffer handle AND type_id, rows, cols, so
# launchers never guess. Asking for the wrong storage kind, or
# an out-of-range layer / suffix, refuses gpu_weights_bad_slot
# by name.
#
# Meta order mirrors the CPU model exactly: index 0 is
# token_embd.weight, index 1 is output.weight, then per-layer
# slots in (layer, suffix_index) order as GlimmerModel.slot
# with LAYER_SUFFIX_COUNT suffixes per layer:
# meta_index(layer, suffix) = 2 + layer * LAYER_SUFFIX_COUNT
# + suffix.

from max.gpu.host import DeviceBuffer

from diag import log_debug
from gpu.context import GpuRuntime, upload_f32, upload_u8

# (Release: build() reserves the exact total on the runtime's
# budget ledger; release_reservation() below returns it.)
from model.model import DT_F32, LAYER_SUFFIX_COUNT, GlimmerModel, TensorSlot


def slot_resident_bytes(type_id: Int, byte_len: Int, value_count: Int) -> Int:
    """One slot's device-resident bytes: F32 slots store
    float32 values (4 B each); every other slot stores its
    raw bytes as-is."""
    if type_id == DT_F32:
        return value_count * 4
    return byte_len


def meta_index(layer: Int, suffix_index: Int) -> Int:
    """DeviceWeights.metas index of a per-layer slot."""
    return 2 + layer * LAYER_SUFFIX_COUNT + suffix_index


def weights_total_bytes(model: GlimmerModel) raises -> Int:
    """The EXACT resident byte total of a model's weights:
    every tensor slot's device bytes plus the final-norm
    values. Computed from the loader's cached figures; no
    device involved, so this is VM-testable and is the number
    the budget reservation is made from."""
    var total = 0
    total += _slot_resident_checked(model.embd.copy())
    total += _slot_resident_checked(model.output.copy())
    total += len(model.final_norm) * 4
    for layer in range(model.cfg.layers):
        for s in range(LAYER_SUFFIX_COUNT):
            total += _slot_resident_checked(model.slot(layer, s))
    return total


def _slot_resident_checked(slot: TensorSlot) raises -> Int:
    if slot.type_id == DT_F32 and len(slot.values) * 4 != len(slot.bytes):
        raise Error(
            "gpu_weights_bad_slot: F32 slot "
            + slot.name
            + " has "
            + String(len(slot.values))
            + " values but "
            + String(len(slot.bytes))
            + " raw bytes"
        )
    return slot_resident_bytes(slot.type_id, len(slot.bytes), len(slot.values))


struct DeviceSlotMeta(Copyable, Movable):
    """One resident slot's facts, no buffer handle."""

    var name: String
    var type_id: Int
    var rows: Int
    var cols: Int
    var resident_bytes: Int
    var is_f32: Bool
    var buffer_index: Int

    def __init__(
        out self,
        name: String,
        type_id: Int,
        rows: Int,
        cols: Int,
        resident_bytes: Int,
        is_f32: Bool,
        buffer_index: Int,
    ):
        self.name = name
        self.type_id = type_id
        self.rows = rows
        self.cols = cols
        self.resident_bytes = resident_bytes
        self.is_f32 = is_f32
        self.buffer_index = buffer_index


struct DeviceSlotU8(Copyable, Movable):
    """A uint8 slot buffer plus the facts a launcher needs."""

    var buffer: DeviceBuffer[DType.uint8]
    var name: String
    var type_id: Int
    var rows: Int
    var cols: Int

    def __init__(
        out self,
        var buffer: DeviceBuffer[DType.uint8],
        name: String,
        type_id: Int,
        rows: Int,
        cols: Int,
    ):
        self.buffer = buffer^
        self.name = name
        self.type_id = type_id
        self.rows = rows
        self.cols = cols


struct DeviceSlotF32(Copyable, Movable):
    """A float32 slot buffer plus the facts a launcher needs."""

    var buffer: DeviceBuffer[DType.float32]
    var name: String
    var type_id: Int
    var rows: Int
    var cols: Int

    def __init__(
        out self,
        var buffer: DeviceBuffer[DType.float32],
        name: String,
        type_id: Int,
        rows: Int,
        cols: Int,
    ):
        self.buffer = buffer^
        self.name = name
        self.type_id = type_id
        self.rows = rows
        self.cols = cols


struct DeviceWeights(Movable):
    """All of a GlimmerModel's weights, resident on device."""

    var metas: List[DeviceSlotMeta]
    var u8_buffers: List[DeviceBuffer[DType.uint8]]
    var f32_buffers: List[DeviceBuffer[DType.float32]]
    var output_norm_buffer: DeviceBuffer[DType.float32]
    var output_norm_meta: DeviceSlotMeta
    var total_resident_bytes: Int
    var layers: Int
    var reservation_released: Bool

    def __init__(
        out self,
        var output_norm_buffer: DeviceBuffer[DType.float32],
        var output_norm_meta: DeviceSlotMeta,
        total_resident_bytes: Int,
        layers: Int,
    ):
        self.metas = List[DeviceSlotMeta]()
        self.u8_buffers = List[DeviceBuffer[DType.uint8]]()
        self.f32_buffers = List[DeviceBuffer[DType.float32]]()
        self.output_norm_buffer = output_norm_buffer^
        self.output_norm_meta = output_norm_meta^
        self.total_resident_bytes = total_resident_bytes
        self.layers = layers
        self.reservation_released = False

    @staticmethod
    def build(mut rt: GpuRuntime, model: GlimmerModel) raises -> DeviceWeights:
        """Reserve the exact total, then upload every slot.

        Raises gpu_budget_exceeded (via the runtime) before
        the first buffer exists when the total is over cap,
        and gpu_weights_bad_slot on an inconsistent F32 slot.
        """
        var total = weights_total_bytes(model)
        rt.reserve_weights(total)
        var norm_meta = DeviceSlotMeta(
            "output_norm.weight",
            DT_F32,
            1,
            len(model.final_norm),
            len(model.final_norm) * 4,
            True,
            -1,
        )
        var norm_buf = upload_f32(rt, model.final_norm)
        var weights = DeviceWeights(norm_buf^, norm_meta^, total, model.cfg.layers)
        weights._add_slot(rt, model.embd.copy())
        weights._add_slot(rt, model.output.copy())
        for layer in range(model.cfg.layers):
            for s in range(LAYER_SUFFIX_COUNT):
                weights._add_slot(rt, model.slot(layer, s))
        _ = log_debug(
            "gpu",
            "device weights built slots="
            + String(len(weights.metas))
            + " resident_bytes="
            + String(weights.total_resident_bytes),
        )
        return weights^

    def _add_slot(mut self, mut rt: GpuRuntime, slot: TensorSlot) raises:
        var is_f32 = slot.type_id == DT_F32
        var resident = slot_resident_bytes(
            slot.type_id, len(slot.bytes), len(slot.values)
        )
        var buffer_index: Int
        if is_f32:
            buffer_index = len(self.f32_buffers)
            var buf = upload_f32(rt, slot.values)
            self.f32_buffers.append(buf^)
        else:
            buffer_index = len(self.u8_buffers)
            var buf2 = upload_u8(rt, slot.bytes)
            self.u8_buffers.append(buf2^)
        self.metas.append(
            DeviceSlotMeta(
                slot.name,
                slot.type_id,
                slot.rows,
                slot.cols,
                resident,
                is_f32,
                buffer_index,
            )
        )

    # -- inventory ---------------------------------------------------

    def slot_count(self) -> Int:
        """Tensor slots resident, final-norm excluded (it has
        its own accessor): 2 + layers * LAYER_SUFFIX_COUNT."""
        return len(self.metas)

    def layer_count(self) -> Int:
        return self.layers

    def resident_bytes(self) -> Int:
        return self.total_resident_bytes

    def release_reservation(mut self, mut rt: GpuRuntime) raises:
        """Return build()'s weights reservation to the runtime
        ledger (Wave E): call when this structure is being torn
        down, before or as its buffers go away, so a later build
        in the same runtime can reserve the same room. Releasing
        twice refuses gpu_budget_exceeded by name (through the
        ledger's own check, and this flag keeps the refusal
        exact even if another weights build reserved in between).
        """
        if self.reservation_released:
            raise Error(
                "gpu_budget_exceeded: weights reservation of "
                + String(self.total_resident_bytes)
                + " B was already released"
            )
        rt.unreserve_weights(self.total_resident_bytes)
        self.reservation_released = True
        _ = log_debug(
            "gpu",
            "device weights reservation released bytes="
            + String(self.total_resident_bytes),
        )

    # -- meta accessors ----------------------------------------------

    def embedding_meta(self) -> DeviceSlotMeta:
        return self.metas[0].copy()

    def output_meta(self) -> DeviceSlotMeta:
        return self.metas[1].copy()

    def final_norm_meta(self) -> DeviceSlotMeta:
        """The output_norm.weight facts (rows 1, cols hidden)."""
        return self.output_norm_meta.copy()

    def slot_meta(self, layer: Int, suffix_index: Int) raises -> DeviceSlotMeta:
        if layer < 0 or layer >= self.layers:
            raise Error(
                "gpu_weights_bad_slot: layer "
                + String(layer)
                + " out of range, model has "
                + String(self.layers)
                + " layers"
            )
        if suffix_index < 0 or suffix_index >= LAYER_SUFFIX_COUNT:
            raise Error(
                "gpu_weights_bad_slot: suffix index "
                + String(suffix_index)
                + " out of range, LAYER_SUFFIX_COUNT is "
                + String(LAYER_SUFFIX_COUNT)
            )
        return self.metas[meta_index(layer, suffix_index)].copy()

    # -- buffer accessors ----------------------------------------------

    def _u8_view(self, meta: DeviceSlotMeta) raises -> DeviceSlotU8:
        if meta.is_f32:
            raise Error(
                "gpu_weights_bad_slot: slot "
                + meta.name
                + " is stored as float32, not uint8"
            )
        return DeviceSlotU8(
            self.u8_buffers[meta.buffer_index].copy(),
            meta.name,
            meta.type_id,
            meta.rows,
            meta.cols,
        )

    def _f32_view(self, meta: DeviceSlotMeta) raises -> DeviceSlotF32:
        if not meta.is_f32:
            raise Error(
                "gpu_weights_bad_slot: slot "
                + meta.name
                + " is stored as uint8 bytes, not float32"
            )
        return DeviceSlotF32(
            self.f32_buffers[meta.buffer_index].copy(),
            meta.name,
            meta.type_id,
            meta.rows,
            meta.cols,
        )

    def embedding_u8(self) raises -> DeviceSlotU8:
        return self._u8_view(self.embedding_meta())

    def embedding_f32(self) raises -> DeviceSlotF32:
        return self._f32_view(self.embedding_meta())

    def output_u8(self) raises -> DeviceSlotU8:
        return self._u8_view(self.output_meta())

    def output_f32(self) raises -> DeviceSlotF32:
        return self._f32_view(self.output_meta())

    def slot_u8(self, layer: Int, suffix_index: Int) raises -> DeviceSlotU8:
        return self._u8_view(self.slot_meta(layer, suffix_index))

    def slot_f32(self, layer: Int, suffix_index: Int) raises -> DeviceSlotF32:
        return self._f32_view(self.slot_meta(layer, suffix_index))

    def output_norm(self) -> DeviceSlotF32:
        """The final-norm values as a float32 device buffer
        (name output_norm.weight, cols = hidden)."""
        return DeviceSlotF32(
            self.output_norm_buffer.copy(),
            self.output_norm_meta.name,
            self.output_norm_meta.type_id,
            self.output_norm_meta.rows,
            self.output_norm_meta.cols,
        )

    def output_norm_values(self) -> DeviceBuffer[DType.float32]:
        """Just the final-norm float32 buffer handle."""
        return self.output_norm_buffer.copy()
