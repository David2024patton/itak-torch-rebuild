// Cross-check harness for the Q2_K / Q3_K dequantizer port.
//
// Dequantizes caller-supplied blocks with ggml's own reference row
// functions (dequantize_row_q2_K / dequantize_row_q3_K) and quantizes
// f32 rows with ggml's reference quantizers (quantize_row_q2_K_ref /
// quantize_row_q3_K_ref), so the Rust port in crates/gguf can be
// compared against the exact code it was ported from, byte for byte.
//
// Build, two equivalent routes:
//   (a) link the tree's built libggml-base (what run-server.sh does):
//       gcc -O2 -std=c11 -I<tree>/ggml/include -I<tree>/ggml/src \
//           harness.c -o harness -L<build>/bin -lggml-base -lm \
//           -Wl,-rpath,<build>/bin
//   (b) compile ggml-quants.c standalone, garbage-collecting the
//       unused quantizers (the four functions used here are
//       self-contained; without --gc-sections the link fails on
//       ggml_abort / ggml_row_size from unrelated IQ quantizers):
//       gcc -O2 -std=c11 -ffunction-sections -fdata-sections \
//           -I<tree>/ggml/include -I<tree>/ggml/src \
//           harness.c <tree>/ggml/src/ggml-quants.c -o harness -lm \
//           -Wl,--gc-sections
//
// Modes:
//   harness dequant q2k|q3k <blocks.bin> <out.f32>
//   harness quant   q2k|q3k <in.f32>    <blocks.bin>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "ggml-quants.h"

static void die(const char * msg) {
    fprintf(stderr, "harness: %s\n", msg);
    exit(2);
}

static uint8_t * read_all(const char * path, long * out_size) {
    FILE * f = fopen(path, "rb");
    if (!f) die("cannot open input");
    if (fseek(f, 0, SEEK_END) != 0) die("seek failed");
    long size = ftell(f);
    if (size < 0) die("tell failed");
    if (fseek(f, 0, SEEK_SET) != 0) die("seek failed");
    uint8_t * buf = malloc(size > 0 ? (size_t) size : 1);
    if (!buf) die("oom");
    if (size > 0 && fread(buf, 1, (size_t) size, f) != (size_t) size) die("read failed");
    fclose(f);
    *out_size = size;
    return buf;
}

static void write_all(const char * path, const void * data, long size) {
    FILE * f = fopen(path, "wb");
    if (!f) die("cannot open output");
    if (size > 0 && fwrite(data, 1, (size_t) size, f) != (size_t) size) die("write failed");
    fclose(f);
}

int main(int argc, char ** argv) {
    if (argc != 5) die("usage: harness <dequant|quant> <q2k|q3k> <in> <out>");
    const int is_q2 = strcmp(argv[2], "q2k") == 0;
    if (!is_q2 && strcmp(argv[2], "q3k") != 0) die("type must be q2k or q3k");

    fprintf(stderr, "sizeof(block_q2_K)=%zu sizeof(block_q3_K)=%zu QK_K=%d\n",
            sizeof(block_q2_K), sizeof(block_q3_K), QK_K);

    long in_size = 0;
    uint8_t * in = read_all(argv[3], &in_size);

    if (strcmp(argv[1], "dequant") == 0) {
        const long block_bytes = is_q2 ? (long) sizeof(block_q2_K) : (long) sizeof(block_q3_K);
        if (in_size % block_bytes != 0) die("input is not a whole number of blocks");
        const long nblocks = in_size / block_bytes;
        float * out = malloc((size_t) (nblocks * QK_K) * sizeof(float));
        if (!out) die("oom");
        if (is_q2) {
            dequantize_row_q2_K((const block_q2_K *) in, out, nblocks * QK_K);
        } else {
            dequantize_row_q3_K((const block_q3_K *) in, out, nblocks * QK_K);
        }
        write_all(argv[4], out, nblocks * QK_K * (long) sizeof(float));
        free(out);
    } else if (strcmp(argv[1], "quant") == 0) {
        const long row_bytes = QK_K * (long) sizeof(float);
        if (in_size % row_bytes != 0) die("input is not a whole number of 256-element rows");
        const long nrows = in_size / row_bytes;
        const long block_bytes = is_q2 ? (long) sizeof(block_q2_K) : (long) sizeof(block_q3_K);
        uint8_t * out = malloc((size_t) (nrows * block_bytes));
        if (!out) die("oom");
        if (is_q2) {
            quantize_row_q2_K_ref((const float *) in, (block_q2_K *) out, nrows * QK_K);
        } else {
            quantize_row_q3_K_ref((const float *) in, (block_q3_K *) out, nrows * QK_K);
        }
        write_all(argv[4], out, nrows * block_bytes);
        free(out);
    } else {
        die("mode must be dequant or quant");
    }

    free(in);
    return 0;
}
