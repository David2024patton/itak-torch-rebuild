# gpu_proof_ops.mojo - server-card proof driver for Lane A4 ops.
#
# NOT a TestSuite file: run explicitly on a GPU host with
#   mojo run -I src tests/gpu_proof_ops.mojo
# Launches rms_norm_kernel and softmax_kernel (the two ops the
# lane names for card proof) on the same small fixtures the CPU
# suites use, prints the device outputs as GPU_PROOF lines, and
# compares them on-host against the CPU twins at the device bar
# (max abs 1e-4: f32 device accumulation vs the f64-accumulated
# CPU twin). Exits nonzero on any mismatch. On a GPU-less host
# DeviceContext creation fails loudly; that is a host fact, not
# a pass.

from std.sys import exit
from max.gpu.host import DeviceContext

from ops.rms_norm import rms_norm_cpu, rms_norm_kernel
from ops.softmax import softmax_cpu, softmax_kernel


def _check(name: String, got: List[Float32], expected: List[Float32]) -> Bool:
    var worst = Float64(0.0)
    for i in range(len(got)):
        var d = Float64(got[i]) - Float64(expected[i])
        if d < 0:
            d = -d
        if d > worst:
            worst = d
    print("GPU_PROOF", name, "max_abs", worst, "bar 1e-4")
    return worst <= 1.0e-4


def main() raises:
    comptime W = 128
    comptime ROWS = 2
    var x = List[Float32](capacity=ROWS * W)
    x.resize(ROWS * W, Float32(0.0))
    var w = List[Float32](capacity=W)
    w.resize(W, Float32(0.0))
    for i in range(ROWS * W):
        x[i] = Float32(((i * 37) % 23) - 11) / Float32(4.0)
    for i in range(W):
        w[i] = Float32(1.0) + Float32(((i * 37 + 77) % 23) - 11) / Float32(32.0)
    var ctx = DeviceContext()
    var ok = True
    # RMS norm, row by row on the CPU twin for the reference.
    var x_dev = ctx.enqueue_create_buffer[DType.float32](ROWS * W)
    var w_dev = ctx.enqueue_create_buffer[DType.float32](W)
    var y_dev = ctx.enqueue_create_buffer[DType.float32](ROWS * W)
    var x_host = ctx.enqueue_create_host_buffer[DType.float32](ROWS * W)
    var w_host = ctx.enqueue_create_host_buffer[DType.float32](W)
    for i in range(ROWS * W):
        x_host[i] = x[i]
    for i in range(W):
        w_host[i] = w[i]
    ctx.enqueue_copy(src_buf=x_host, dst_buf=x_dev)
    ctx.enqueue_copy(src_buf=w_host, dst_buf=w_dev)
    ctx.enqueue_function[rms_norm_kernel](
        x_dev,
        w_dev,
        y_dev,
        Int32(W),
        Float32(1.0e-5),
        grid_dim=ROWS,
        block_dim=256,
    )
    ctx.synchronize()
    var y_host = ctx.enqueue_create_host_buffer[DType.float32](ROWS * W)
    ctx.enqueue_copy(src_buf=y_dev, dst_buf=y_host)
    ctx.synchronize()
    var got = List[Float32](capacity=ROWS * W)
    got.resize(ROWS * W, Float32(0.0))
    for i in range(ROWS * W):
        got[i] = y_host[i]
    for r in range(ROWS):
        var row = List[Float32](capacity=W)
        row.resize(W, Float32(0.0))
        var grow = List[Float32](capacity=W)
        grow.resize(W, Float32(0.0))
        for i in range(W):
            row[i] = x[r * W + i]
            grow[i] = got[r * W + i]
        var expct = rms_norm_cpu(row, w, Float32(1.0e-5))
        if not _check("rms_norm row " + String(r), grow, expct):
            ok = False
    # Softmax over the same buffer as 2 rows of scores.
    var s_dev = ctx.enqueue_create_buffer[DType.float32](ROWS * W)
    ctx.enqueue_copy(src_buf=x_host, dst_buf=s_dev)
    var o_dev = ctx.enqueue_create_buffer[DType.float32](ROWS * W)
    ctx.enqueue_function[softmax_kernel](
        s_dev, o_dev, Int32(W), grid_dim=ROWS, block_dim=256
    )
    ctx.synchronize()
    var o_host = ctx.enqueue_create_host_buffer[DType.float32](ROWS * W)
    ctx.enqueue_copy(src_buf=o_dev, dst_buf=o_host)
    ctx.synchronize()
    var sgot = List[Float32](capacity=ROWS * W)
    sgot.resize(ROWS * W, Float32(0.0))
    for i in range(ROWS * W):
        sgot[i] = o_host[i]
    var sexp = softmax_cpu(x, W)
    if not _check("softmax 2x128", sgot, sexp):
        ok = False
    if not ok:
        print("GPU_PROOF FAIL")
        exit(1)
    print("GPU_PROOF PASS")
