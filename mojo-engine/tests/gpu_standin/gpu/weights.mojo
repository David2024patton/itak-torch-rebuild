# weights.mojo - STAND-IN DeviceWeights for Lane C2's proof.
#
# Minimal stand-in for Lane C1's src/gpu/weights.mojo (see
# tests/gpu_standin/gpu/__init__.mojo). Built from a
# GlimmerModel's slots; every slot's RAW BYTES are uploaded
# once as a uint8 device buffer (quantized and F16 slots are
# byte layouts; F32 slots also carry raw bytes in the CPU
# TensorSlot, and Lane C2 reads F32 norm values from the CPU
# model, never from this buffer). Accessors return
# (buffer, type_id, rows, cols) so launchers never guess.
#
# Surface Lane C2 uses:
#   embedding_u8() -> (buffer, type_id, rows, cols)
#   output_u8()    -> (buffer, type_id, rows, cols)
#   layer_u8(layer, suffix_index) -> (buffer, type_id, rows, cols)
#
# Named refusal: gpu_weights_bad_slot.

from max.gpu.host import DeviceBuffer

from diag import log_debug
from gpu.context import GpuRuntime, upload_u8
from model.model import GlimmerModel, LAYER_SUFFIX_COUNT


struct DeviceWeights(Movable):
    var embd_buf: DeviceBuffer[DType.uint8]
    var embd_type: Int
    var embd_rows: Int
    var embd_cols: Int
    var out_buf: DeviceBuffer[DType.uint8]
    var out_type: Int
    var out_rows: Int
    var out_cols: Int
    var layer_bufs: List[DeviceBuffer[DType.uint8]]
    var layer_types: List[Int]
    var layer_rows: List[Int]
    var layer_cols: List[Int]
    var resident_bytes: Int

    def __init__(
        out self,
        var embd_buf: DeviceBuffer[DType.uint8],
        embd_type: Int,
        embd_rows: Int,
        embd_cols: Int,
        var out_buf: DeviceBuffer[DType.uint8],
        out_type: Int,
        out_rows: Int,
        out_cols: Int,
        var layer_bufs: List[DeviceBuffer[DType.uint8]],
        var layer_types: List[Int],
        var layer_rows: List[Int],
        var layer_cols: List[Int],
        resident_bytes: Int,
    ):
        self.embd_buf = embd_buf^
        self.embd_type = embd_type
        self.embd_rows = embd_rows
        self.embd_cols = embd_cols
        self.out_buf = out_buf^
        self.out_type = out_type
        self.out_rows = out_rows
        self.out_cols = out_cols
        self.layer_bufs = layer_bufs^
        self.layer_types = layer_types^
        self.layer_rows = layer_rows^
        self.layer_cols = layer_cols^
        self.resident_bytes = resident_bytes

    @staticmethod
    def from_model(
        rt: GpuRuntime, model: GlimmerModel
    ) raises -> DeviceWeights:
        """Stand-in construction: upload every slot's bytes
        once, in the CPU model's own slot order."""
        var resident = 0
        var embd_buf = upload_u8(rt, model.embd.bytes)
        resident += len(model.embd.bytes)
        var out_buf = upload_u8(rt, model.output.bytes)
        resident += len(model.output.bytes)
        var layer_bufs = List[DeviceBuffer[DType.uint8]]()
        var layer_types = List[Int]()
        var layer_rows = List[Int]()
        var layer_cols = List[Int]()
        for i in range(len(model.layer_weights)):
            var slot = model.layer_weights[i].copy()
            layer_bufs.append(upload_u8(rt, slot.bytes))
            layer_types.append(slot.type_id)
            layer_rows.append(slot.rows)
            layer_cols.append(slot.cols)
            resident += len(slot.bytes)
        _ = log_debug(
            "gpu.standin",
            "DeviceWeights stand-in resident bytes=" + String(resident),
        )
        return DeviceWeights(
            embd_buf^,
            model.embd.type_id,
            model.embd.rows,
            model.embd.cols,
            out_buf^,
            model.output.type_id,
            model.output.rows,
            model.output.cols,
            layer_bufs^,
            layer_types^,
            layer_rows^,
            layer_cols^,
            resident,
        )

    def embedding_u8(
        self,
    ) -> Tuple[DeviceBuffer[DType.uint8], Int, Int, Int]:
        return (self.embd_buf.copy(), self.embd_type, self.embd_rows, self.embd_cols)

    def output_u8(
        self,
    ) -> Tuple[DeviceBuffer[DType.uint8], Int, Int, Int]:
        return (self.out_buf.copy(), self.out_type, self.out_rows, self.out_cols)

    def layer_u8(
        self, layer: Int, suffix_index: Int
    ) raises -> Tuple[DeviceBuffer[DType.uint8], Int, Int, Int]:
        if layer < 0 or suffix_index < 0 or suffix_index >= LAYER_SUFFIX_COUNT:
            raise Error(
                "gpu_weights_bad_slot: layer "
                + String(layer)
                + " suffix "
                + String(suffix_index)
                + " out of range"
            )
        var idx = layer * LAYER_SUFFIX_COUNT + suffix_index
        if idx >= len(self.layer_bufs):
            raise Error(
                "gpu_weights_bad_slot: layer "
                + String(layer)
                + " suffix "
                + String(suffix_index)
                + " beyond uploaded slots "
                + String(len(self.layer_bufs))
            )
        return (
            self.layer_bufs[idx].copy(),
            self.layer_types[idx],
            self.layer_rows[idx],
            self.layer_cols[idx],
        )
