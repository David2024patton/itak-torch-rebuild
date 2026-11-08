# weights.mojo - STAND-IN DeviceWeights mirroring C1's
# surface (see tests/gpu_standin/gpu/__init__.mojo).
#
# Same structs and accessor names as src/gpu/weights.mojo
# (Lane C1): DeviceSlotMeta / DeviceSlotU8 / DeviceSlotF32,
# DeviceWeights.build(mut rt, model), embedding_meta /
# output_meta / final_norm_meta / slot_meta, embedding_u8 /
# output_u8 / slot_u8 / slot_f32 / output_f32 /
# embedding_f32 / output_norm / output_norm_values,
# resident_bytes(). Meta order mirrors C1: index 0 embedding,
# 1 output, then per-layer (layer, suffix_index).

from max.gpu.host import DeviceBuffer

from diag import log_debug
from gpu.context import GpuRuntime, upload_f32, upload_u8
from model.model import DT_F32, LAYER_SUFFIX_COUNT, GlimmerModel, TensorSlot


def meta_index(layer: Int, suffix_index: Int) -> Int:
    return 2 + layer * LAYER_SUFFIX_COUNT + suffix_index


def _slot_resident(slot: TensorSlot) -> Int:
    if slot.type_id == DT_F32:
        return len(slot.values) * 4
    return len(slot.bytes)


struct DeviceSlotMeta(Copyable, Movable):
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
    var metas: List[DeviceSlotMeta]
    var u8_buffers: List[DeviceBuffer[DType.uint8]]
    var f32_buffers: List[DeviceBuffer[DType.float32]]
    var output_norm_buffer: DeviceBuffer[DType.float32]
    var output_norm_meta: DeviceSlotMeta
    var total_resident_bytes: Int
    var layers: Int

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

    @staticmethod
    def build(mut rt: GpuRuntime, model: GlimmerModel) raises -> DeviceWeights:
        var total = len(model.final_norm) * 4
        total += _slot_resident(model.embd.copy())
        total += _slot_resident(model.output.copy())
        for layer in range(model.cfg.layers):
            for s in range(LAYER_SUFFIX_COUNT):
                total += _slot_resident(model.slot(layer, s))
        rt.reserve_weights(total)
        var norm_meta = DeviceSlotMeta(
            "output_norm.weight", DT_F32, 1, len(model.final_norm),
            len(model.final_norm) * 4, True, -1,
        )
        var norm_buf = upload_f32(rt, model.final_norm)
        var weights = DeviceWeights(norm_buf^, norm_meta^, total, model.cfg.layers)
        weights._add_slot(rt, model.embd.copy())
        weights._add_slot(rt, model.output.copy())
        for layer in range(model.cfg.layers):
            for s in range(LAYER_SUFFIX_COUNT):
                weights._add_slot(rt, model.slot(layer, s))
        _ = log_debug(
            "gpu.standin", "DeviceWeights stand-in resident=" + String(total)
        )
        return weights^

    def _add_slot(mut self, mut rt: GpuRuntime, slot: TensorSlot) raises:
        var is_f32 = slot.type_id == DT_F32
        var buffer_index = (
            len(self.f32_buffers) if is_f32 else len(self.u8_buffers)
        )
        if is_f32:
            var buf = upload_f32(rt, slot.values)
            self.f32_buffers.append(buf^)
        else:
            var buf2 = upload_u8(rt, slot.bytes)
            self.u8_buffers.append(buf2^)
        self.metas.append(
            DeviceSlotMeta(
                slot.name, slot.type_id, slot.rows, slot.cols,
                _slot_resident(slot), is_f32, buffer_index,
            )
        )

    def slot_count(self) -> Int:
        return len(self.metas)

    def layer_count(self) -> Int:
        return self.layers

    def resident_bytes(self) -> Int:
        return self.total_resident_bytes

    def embedding_meta(self) -> DeviceSlotMeta:
        return self.metas[0].copy()

    def output_meta(self) -> DeviceSlotMeta:
        return self.metas[1].copy()

    def final_norm_meta(self) -> DeviceSlotMeta:
        return self.output_norm_meta.copy()

    def slot_meta(self, layer: Int, suffix_index: Int) raises -> DeviceSlotMeta:
        if layer < 0 or layer >= self.layers:
            raise Error(
                "gpu_weights_bad_slot: layer " + String(layer) + " out of range"
            )
        if suffix_index < 0 or suffix_index >= LAYER_SUFFIX_COUNT:
            raise Error(
                "gpu_weights_bad_slot: suffix " + String(suffix_index)
                + " out of range"
            )
        return self.metas[meta_index(layer, suffix_index)].copy()

    def _u8_view(self, meta: DeviceSlotMeta) raises -> DeviceSlotU8:
        if meta.is_f32:
            raise Error(
                "gpu_weights_bad_slot: slot " + meta.name
                + " is stored as float32, not uint8"
            )
        return DeviceSlotU8(
            self.u8_buffers[meta.buffer_index].copy(), meta.name,
            meta.type_id, meta.rows, meta.cols,
        )

    def _f32_view(self, meta: DeviceSlotMeta) raises -> DeviceSlotF32:
        if not meta.is_f32:
            raise Error(
                "gpu_weights_bad_slot: slot " + meta.name
                + " is stored as uint8 bytes, not float32"
            )
        return DeviceSlotF32(
            self.f32_buffers[meta.buffer_index].copy(), meta.name,
            meta.type_id, meta.rows, meta.cols,
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
        return DeviceSlotF32(
            self.output_norm_buffer.copy(), self.output_norm_meta.name,
            self.output_norm_meta.type_id, self.output_norm_meta.rows,
            self.output_norm_meta.cols,
        )

    def output_norm_values(self) -> DeviceBuffer[DType.float32]:
        return self.output_norm_buffer.copy()
