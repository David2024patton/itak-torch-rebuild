// cb_dump: per-layer activation dump from llama.cpp (layer-diff
// harness, follow-up to integration run A).
//
// Loads a GGUF model on the CPU backend only, runs ONE prefill of
// fixed token ids through llama_decode with an eval callback
// (llama_context_params.cb_eval) attached, and writes every named
// graph intermediate the arch-llama side can also produce
// (crates/arch-llama/examples/layer_dump.rs) as raw little-endian
// f32 .bin files plus a manifest.json of shapes. Names are mapped
// to the shared scheme:
//
//   inp_embd        -> embd            attn_norm-<il> -> L<il>_attn_norm
//   Qcur-<il>       -> L<il>_q         Kcur-<il>      -> L<il>_k
//   Vcur-<il>       -> L<il>_v         ffn_inp-<il>   -> L<il>_ffn_inp
//   ffn_norm-<il>   -> L<il>_ffn_norm  ffn_out-<il>   -> L<il>_ffn_out
//   l_out-<il>      -> L<il>_l_out     result_norm--1 -> result_norm
//   result_output--1 -> logits
//
// ggml tensors are [ne0 = width, ne1 = rows]; rows are copied out
// honoring nb[1], so the files are flat [rows][width] f32, the same
// layout the Rust dumper writes.
//
// Usage: cb_dump <model.gguf> <out_dir> <id,id,...>
// Build (on the server, against its llama.cpp tree):
//   g++ -O2 -std=c++17 -I~/llama.cpp/include -I~/llama.cpp/ggml/include \
//       cb_dump.cpp -o cb_dump -L~/llama.cpp/build-vulkan2/bin \
//       -lllama -lggml -lggml-base -lggml-cpu \
//       -Wl,-rpath,/home/server/llama.cpp/build-vulkan2/bin

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <sys/stat.h>

static std::string g_outdir;
static bool g_armed = false;
static FILE * g_manifest = nullptr;
static bool g_first_entry = true;

// Map a graph tensor name to the shared file stem, or "" to skip.
static std::string map_name(const char * name) {
    const std::string n(name);
    if (n == "inp_embd" || n == "embd") return "embd";
    if (n.rfind("result_norm", 0) == 0) return "result_norm";
    if (n.rfind("result_output", 0) == 0) return "logits";
    static const struct { const char * base; const char * stem; } kLayers[] = {
        { "attn_norm-", "_attn_norm" },
        { "Qcur-",      "_q" },
        { "Kcur-",      "_k" },
        { "Vcur-",      "_v" },
        { "ffn_inp-",   "_ffn_inp" },
        { "ffn_norm-",  "_ffn_norm" },
        { "ffn_out-",   "_ffn_out" },
        { "l_out-",     "_l_out" },
    };
    for (const auto & e : kLayers) {
        const size_t blen = strlen(e.base);
        if (n.rfind(e.base, 0) == 0) {
            const int il = atoi(n.c_str() + blen);
            char buf[64];
            snprintf(buf, sizeof(buf), "L%02d%s", il, e.stem);
            return buf;
        }
    }
    return "";
}

static bool cb_eval(ggml_tensor * t, bool ask, void * user_data) {
    (void) user_data;
    if (!g_armed || t == nullptr || t->name[0] == '\0') {
        return false;
    }
    const std::string stem = map_name(t->name);
    if (stem.empty()) {
        return false;
    }
    if (ask) {
        return true; // ask the scheduler to hand us the computed data
    }
    if (t->type != GGML_TYPE_F32) {
        fprintf(stderr, "cb_dump: skip %s (type %s, not F32)\n", t->name, ggml_type_name(t->type));
        return true;
    }
    // Flatten in ggml element order (ne[0] fastest): for a
    // [width, rows] tensor that is exactly [rows][width] row-major,
    // and for [head_dim, heads, rows] (Qcur/Kcur/Vcur) it is
    // [rows][heads][head_dim], which is the Rust layout too.
    const int64_t ne0 = t->ne[0];
    std::vector<float> buf((size_t) ggml_nelements(t));
    size_t dst = 0;
    for (int64_t i3 = 0; i3 < t->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < t->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < t->ne[1]; ++i1) {
                const size_t off = (size_t) i1 * t->nb[1] + (size_t) i2 * t->nb[2]
                                 + (size_t) i3 * t->nb[3];
                ggml_backend_tensor_get(t, buf.data() + dst, off, (size_t) (ne0 * 4));
                dst += (size_t) ne0;
            }
        }
    }
    const std::string path = g_outdir + "/" + stem + ".bin";
    FILE * f = fopen(path.c_str(), "wb");
    if (f == nullptr) {
        fprintf(stderr, "cb_dump: cannot write %s\n", path.c_str());
        return true;
    }
    fwrite(buf.data(), 4, buf.size(), f);
    fclose(f);
    if (g_manifest != nullptr) {
        fprintf(g_manifest, "%s  \"%s\": [%lld", g_first_entry ? "" : "", stem.c_str(),
                (long long) t->ne[0]);
        for (int d = 1; d < ggml_n_dims(t); ++d) {
            fprintf(g_manifest, ", %lld", (long long) t->ne[d]);
        }
        fprintf(g_manifest, "],\n");
        g_first_entry = false;
        fflush(g_manifest);
    }
    return true;
}

int main(int argc, char ** argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: cb_dump <model.gguf> <out_dir> <id,id,...>\n");
        return 1;
    }
    const char * model_path = argv[1];
    g_outdir = argv[2];
    std::vector<llama_token> ids;
    {
        std::string s(argv[3]);
        size_t at = 0;
        while (at < s.size()) {
            const size_t comma = s.find(',', at);
            ids.push_back((llama_token) atoi(s.substr(at, comma - at).c_str()));
            if (comma == std::string::npos) break;
            at = comma + 1;
        }
    }
    if (ids.empty()) {
        fprintf(stderr, "cb_dump: no ids\n");
        return 1;
    }
    mkdir(g_outdir.c_str(), 0755);

    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0; // CPU only: the production GPU server is not ours to touch
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (model == nullptr) {
        fprintf(stderr, "cb_dump: model load failed\n");
        return 1;
    }
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 128;
    cparams.n_batch = 128;
    cparams.no_perf = true;
    cparams.cb_eval = cb_eval;
    cparams.cb_eval_user_data = nullptr;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        fprintf(stderr, "cb_dump: context init failed\n");
        return 1;
    }

    const std::string mpath = g_outdir + "/manifest.json";
    g_manifest = fopen(mpath.c_str(), "w");
    fprintf(g_manifest, "{\n");

    llama_batch batch = llama_batch_init((int32_t) ids.size(), 0, 1);
    for (size_t i = 0; i < ids.size(); ++i) {
        batch.token[i] = ids[i];
        batch.pos[i] = (llama_pos) i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 1; // logits on every position: keeps the last layer unsliced
    }
    batch.n_tokens = (int32_t) ids.size();

    g_armed = true;
    const int rc = llama_decode(ctx, batch);
    g_armed = false;
    if (rc != 0) {
        fprintf(stderr, "cb_dump: llama_decode failed (%d)\n", rc);
        return 1;
    }

    // Sanity: the last position's top-1 from the public logits API.
    const float * logits = llama_get_logits_ith(ctx, -1);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    int top = 0;
    for (int i = 1; i < n_vocab; ++i) {
        if (logits[i] > logits[top]) top = i;
    }
    fprintf(stderr, "cb_dump: %zu tokens, top-1 logit id %d (%.4f)\n", ids.size(), top, logits[top]);

    if (g_manifest != nullptr) {
        // Trim the trailing comma of the last entry: rewrite the tail.
        fclose(g_manifest);
        g_manifest = nullptr;
        std::string text;
        FILE * f = fopen(mpath.c_str(), "rb");
        char chunk[4096];
        size_t n;
        while ((n = fread(chunk, 1, sizeof(chunk), f)) > 0) text.append(chunk, n);
        fclose(f);
        const size_t comma = text.rfind(",\n");
        if (comma != std::string::npos) text.replace(comma, 2, "\n");
        text += "}\n";
        f = fopen(mpath.c_str(), "w");
        fwrite(text.data(), 1, text.size(), f);
        fclose(f);
    }

    FILE * meta = fopen((g_outdir + "/meta.txt").c_str(), "w");
    fprintf(meta, "engine=llama.cpp-cb_eval n_vocab=%d ids=", n_vocab);
    for (size_t i = 0; i < ids.size(); ++i) fprintf(meta, "%s%d", i ? "," : "", ids[i]);
    fprintf(meta, "\n");
    fclose(meta);

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
