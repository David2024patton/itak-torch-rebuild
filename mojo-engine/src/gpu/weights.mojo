# weights.mojo - DeviceWeights: every tensor slot resident on
# device, uploaded once at load (Lane C1).
#
# =====================================================================
# PLACEHOLDER / STAND-IN (Lane C3, 2026-11-07). This file exists
# in the lane/mojo-c3-gen worktree ONLY so Lane C3's
# src/model/gpu_model.mojo compiles against the Wave C contract
# (mojo-engine/WAVE_C_INTERFACES.md) before Lane C1 lands.
# DELETE AT INTEGRATION: the coordinator replaces this whole
# file with Lane C1's real src/gpu/weights.mojo. Do not ship
# this stand-in as the C1 implementation.
# =====================================================================
#
# Contract surface coded against (WAVE_C_INTERFACES.md, Lane C1):
#   DeviceWeights is built from a GlimmerModel's slots (the CPU
#   model stays the loader of record; its cached bytes are the
#   upload source). One device buffer per tensor slot:
#   quantized and F16 slots as uint8 byte buffers over the
#   slot's raw bytes, F32 slots as float32 buffers over the
#   slot's values. The embedding slot, the output (head) slot,
#   and the final-norm values get named accessors; per-layer
#   slots are reached by (layer, suffix_index) exactly as
#   GlimmerModel.slot. Accessors return the buffer AND the
#   slot's type_id, rows, cols so launchers never guess. The
#   exact resident byte total is accumulated during
#   construction and checked against the budget before the
#   last buffer is created (refuse first, allocate second).
#
# Stand-in shape notes (where this file had to pick a spelling
# the contract leaves open; the coordinator reconciles):
#   - Quantized/F16 buffers live in `byte_buffers`, F32 buffers
#     in `f32_buffers`, in EXACTLY GlimmerModel's slot order:
#     index 0 = token_embd.weight, 1 = output.weight,
#     2 + layer * LAYER_SUFFIX_COUNT + suffix_index = the
#     per-layer slots in model.mojo's layer_weights order.
#   - Accessors return a DeviceSlotView (buffer kind tag +
#     parallel-list index + type_id + rows + cols); the caller
#     reads the buffer out of the matching list by index.
#     DeviceBuffer values are device handles; handing the
#     index keeps one owner (this struct) for every buffer.
#   - Named refusals mirror model.mojo's where a shape is the
#     caller's fault in the same way (gpu_weights_bad_layer,
#     gpu_weights_no_buffer here; C1's real names win at
#     integration if they differ).

from max.gpu.host import DeviceBuffer

from diag import log_debug
from gpu.context import GpuRuntime, upload_f32, upload_u8
from model.model import (
    DT_F32,
    LAYER_SUFFIX_COUNT,
    GlimmerModel,
    TensorSlot,
)

comptime SLOT_EMBD = 0
comptime SLOT_OUTPUT = 1
comptime SLOT_FIRST_LAYER = 2


struct DeviceSlotView(Copyable, Movable):
    """What an accessor returns: which buffer list, which index,
    and the slot's type_id / rows / cols (never guessed)."""

    var is_f32: Bool
    var index: Int
    var type_id: Int
    var rows: Int
    var cols: Int

    def __init__(
        out self, is_f32: Bool, index: Int, type_id: Int, rows: Int, cols: Int
    ):
        self.is_f32 = is_f32
        self.index = index
        self.type_id = type_id
        self.rows = rows
        self.cols = cols


struct DeviceWeights(Movable):
    var byte_buffers: List[DeviceBuffer[DType.uint8]]
    var f32_buffers: List[DeviceBuffer[DType.float32]]
    var views: List[DeviceSlotView]
    var final_norm: DeviceBuffer[DType.float32]
    var total_bytes: Int

    def __init__(
        out self, mut rt: GpuRuntime, model: GlimmerModel
    ) raises:
        self.byte_buffers = List[DeviceBuffer[DType.uint8]]()
        self.f32_buffers = List[DeviceBuffer[DType.float32]]()
        self.views = List[DeviceSlotView]()

        # Refuse first, allocate second: the resident total is
        # computable from the loader's cached bytes up front.
        var total = 0
        total += Self._slot_bytes(model.embd.copy())
        total += Self._slot_bytes(model.output.copy())
        total += len(model.final_norm) * 4
        for i in range(len(model.layer_weights)):
            total += Self._slot_bytes(model.layer_weights[i].copy())
        rt.account(total, "device weights")
        self.total_bytes = total

        self._upload_slot(rt, model.embd.copy())
        self._upload_slot(rt, model.output.copy())
        self.final_norm = upload_f32(rt, model.final_norm.copy())
        for layer in range(model.cfg.layers):
            for s in range(LAYER_SUFFIX_COUNT):
                self._upload_slot(rt, model.slot(layer, s))
        _ = log_debug(
            "gpu",
            "device weights uploaded slots="
            + String(len(self.views))
            + " resident_bytes="
            + String(self.total_bytes),
        )

    @staticmethod
    def _slot_bytes(slot: TensorSlot) -> Int:
        if slot.type_id == DT_F32:
            return len(slot.values) * 4
        return len(slot.bytes)

    def _upload_slot(mut self, rt: GpuRuntime, slot: TensorSlot) raises:
        if slot.type_id == DT_F32:
            var buf = upload_f32(rt, slot.values.copy())
            self.f32_buffers.append(buf^)
            self.views.append(
                DeviceSlotView(
                    True,
                    len(self.f32_buffers) - 1,
                    slot.type_id,
                    slot.rows,
                    slot.cols,
                )
            )
        else:
            var buf = upload_u8(rt, slot.bytes.copy())
            self.byte_buffers.append(buf^)
            self.views.append(
                DeviceSlotView(
                    False,
                    len(self.byte_buffers) - 1,
                    slot.type_id,
                    slot.rows,
                    slot.cols,
                )
            )

    def slot_count(self) -> Int:
        return len(self.views)

    def _view(self, flat: Int) raises -> DeviceSlotView:
        if flat < 0 or flat >= len(self.views):
            raise Error(
                "gpu_weights_no_buffer: flat slot "
                + String(flat)
                + " outside 0.."
                + String(len(self.views) - 1)
            )
        return self.views[flat].copy()

    def embedding_slot(self) raises -> DeviceSlotView:
        """The token_embd.weight slot view."""
        return self._view(SLOT_EMBD)

    def output_slot(self) raises -> DeviceSlotView:
        """The output.weight (head) slot view."""
        return self._view(SLOT_OUTPUT)

    def layer_slot(
        self, layer: Int, suffix_index: Int
    ) raises -> DeviceSlotView:
        """A per-layer slot view by (layer, suffix_index),
        exactly GlimmerModel.slot's addressing."""
        if layer < 0 or suffix_index < 0 or suffix_index >= LAYER_SUFFIX_COUNT:
            raise Error(
                "gpu_weights_bad_layer: layer "
                + String(layer)
                + " suffix "
                + String(suffix_index)
                + " out of range"
            )
        return self._view(SLOT_FIRST_LAYER + layer * LAYER_SUFFIX_COUNT + suffix_index)
