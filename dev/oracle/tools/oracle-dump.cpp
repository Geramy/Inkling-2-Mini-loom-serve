// Dump per-node tensors of the first forward pass of a prompt (f32 / i32 raw .bin + manifest.json).
// Env: ORACLE_TOKENS=file with space separated token ids, ORACLE_OUT=output dir, ORACLE_MODE=all|light
#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <sys/stat.h>

struct dump_state {
    std::string out;
    bool light = false;
    bool off = false;
    int seq = 0;
    std::vector<std::string> entries;
    std::map<std::string, int> occ;
    size_t bytes = 0;
};

static bool wanted(const dump_state & st, const ggml_tensor * t) {
    if (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_I32) {
        return false;
    }
    const std::string n = t->name;
    if (n.find(" (") != std::string::npos) {
        return false;
    }
    if (st.light) {
        return n.rfind("l_out-", 0) == 0 || n == "result_norm" || n == "result_output";
    }
    if (n.empty() || n.rfind("node_", 0) == 0 || n.rfind("leaf_", 0) == 0) {
        return false;
    }
    if (n.rfind("inp_", 0) == 0 && n != "inp_tokens" && n != "inp_embd") {
        return false;
    }
    if (n.find(" (") != std::string::npos || n.rfind("cache_", 0) == 0) {
        return false;
    }
    for (const char * skip : { "ffn_moe_weighted", "ffn_moe_gate", "ffn_moe_up", "ffn_moe_situ", "ffn_moe_down", "kq_soft_max", "kq-" }) {
        if (n.rfind(skip, 0) == 0) {
            return false;
        }
    }
    return ggml_nbytes(t) <= ((size_t) 256 << 20);
}

static std::string json_esc(const std::string & s) {
    std::string r;
    for (char c : s) {
        if (c == '"' || c == 0x5c) {
            r += (char) 0x5c;
        }
        if ((unsigned char) c >= 32) {
            r += c;
        }
    }
    return r;
}

static bool cb_eval(struct ggml_tensor * t, bool ask, void * ud) {
    auto & st = *(dump_state *) ud;
    if (st.off) {
        return ask ? false : true;
    }
    if (ask) {
        return wanted(st, t);
    }
    if (!wanted(st, t)) {
        return true;
    }

    const size_t nb = ggml_nbytes(t);
    std::vector<char> buf;
    const char * data;
    if (ggml_backend_buffer_is_host(t->buffer)) {
        data = (const char *) t->data;
    } else {
        buf.resize(nb);
        ggml_backend_tensor_get(t, buf.data(), 0, nb);
        data = buf.data();
    }

    // logical copy in ggml order (ne0 fastest)
    const size_t es = ggml_type_size(t->type);
    std::vector<char> cont(ggml_nelements(t) * es);
    size_t o = 0;
    for (int64_t i3 = 0; i3 < t->ne[3]; i3++) {
        for (int64_t i2 = 0; i2 < t->ne[2]; i2++) {
            for (int64_t i1 = 0; i1 < t->ne[1]; i1++) {
                for (int64_t i0 = 0; i0 < t->ne[0]; i0++) {
                    memcpy(cont.data() + o, data + i0*t->nb[0] + i1*t->nb[1] + i2*t->nb[2] + i3*t->nb[3], es);
                    o += es;
                }
            }
        }
    }

    const std::string name = t->name;
    int il = -1;
    std::string base = name;
    const size_t dash = name.rfind('-');
    if (dash != std::string::npos && dash + 1 < name.size() && isdigit((unsigned char) name[dash + 1])) {
        il = atoi(name.c_str() + dash + 1);
        base = name.substr(0, dash);
    }
    const int occ = st.occ[name]++;

    char fn[512];
    snprintf(fn, sizeof fn, "%05d_%s.bin", st.seq, name.c_str());
    for (char * p = fn; *p; ++p) {
        if (*p == '/' || *p == '(' || *p == ')' || *p == ' ') {
            *p = '_';
        }
    }
    std::ofstream f(st.out + "/" + fn, std::ios::binary);
    f.write(cont.data(), cont.size());
    st.bytes += cont.size();

    char line[2048];
    snprintf(line, sizeof line,
        "    {\"seq\": %d, \"name\": \"%s\", \"base\": \"%s\", \"layer\": %d, \"occurrence\": %d, \"op\": \"%s\", \"dtype\": \"%s\", \"ne\": [%lld, %lld, %lld, %lld], \"file\": \"%s\"}",
        st.seq, json_esc(name).c_str(), json_esc(base).c_str(), il, occ, ggml_op_name(t->op), t->type == GGML_TYPE_F32 ? "f32" : "i32",
        (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3], fn);
    st.entries.push_back(line);
    st.seq++;
    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    const char * tok_path = getenv("ORACLE_TOKENS");
    const char * out      = getenv("ORACLE_OUT");
    const char * mode     = getenv("ORACLE_MODE");
    if (!tok_path || !out) {
        fprintf(stderr, "set ORACLE_TOKENS and ORACLE_OUT\n");
        return 1;
    }

    dump_state st;
    st.out   = out;
    st.light = mode && std::string(mode) == "light";
    st.off   = mode && std::string(mode) == "none";
    mkdir(out, 0755);

    std::vector<llama_token> tokens;
    {
        std::ifstream f(tok_path);
        long long v;
        while (f >> v) {
            tokens.push_back((llama_token) v);
        }
    }
    if (tokens.empty()) {
        fprintf(stderr, "no tokens\n");
        return 1;
    }

    llama_backend_init();
    params.cb_eval           = cb_eval;
    params.cb_eval_user_data = &st;
    params.warmup            = false;
    params.n_ubatch          = std::max<int>(params.n_ubatch, tokens.size());
    params.n_batch           = std::max<int>(params.n_batch, tokens.size());
    params.n_ctx             = std::max<int>(params.n_ctx, tokens.size() + 8);

    auto init  = common_init_from_params(params);
    auto * model = init->model();
    auto * ctx   = init->context();
    if (!model || !ctx) {
        fprintf(stderr, "init failed\n");
        return 1;
    }

    common_batch batch = common_batch_get_one(ctx, tokens);
    if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
        fprintf(stderr, "decode failed\n");
        return 1;
    }

    std::string meta = "{";
    const int nm = llama_model_meta_count(model);
    bool first = true;
    for (int i = 0; i < nm; i++) {
        char k[256], v[256];
        llama_model_meta_key_by_index(model, i, k, sizeof k);
        llama_model_meta_val_str_by_index(model, i, v, sizeof v);
        if (strlen(v) > 100) {
            continue;
        }
        meta += std::string(first ? "" : ", ") + "\"" + json_esc(k) + "\": \"" + json_esc(v) + "\"";
        first = false;
    }
    meta += "}";
    st.off = true;

    if (const char * gen = getenv("ORACLE_GEN")) {
        const int n_gen = atoi(gen);
        const llama_vocab * vocab = llama_model_get_vocab(model);
        const int n_vocab = llama_vocab_n_tokens(vocab);
        std::vector<llama_token> out_tokens;
        std::string text;
        for (int i = 0; i < n_gen; i++) {
            const float * lg = llama_get_logits_ith(ctx, -1);
            llama_token best = 0;
            for (int j = 1; j < n_vocab; j++) {
                if (lg[j] > lg[best]) {
                    best = j;
                }
            }
            out_tokens.push_back(best);
            text += common_token_to_piece(ctx, best, true);
            if (llama_vocab_is_eog(vocab, best)) {
                break;
            }
            std::vector<llama_token> one = { best };
            common_batch b1 = common_batch_get_one(ctx, one);
            if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, b1.get())) {
                fprintf(stderr, "decode failed\n");
                return 1;
            }
        }
        std::ofstream gt(st.out + "/gen_ids.txt");
        for (size_t i = 0; i < out_tokens.size(); i++) {
            gt << (i ? " " : "") << out_tokens[i];
        }
        gt << std::endl;
        std::ofstream gx(st.out + "/gen_text.txt");
        gx << text;
    }

    std::ofstream m(st.out + "/manifest.json");
    m << "{\n  \"n_tokens\": " << tokens.size() << ",\n  \"tokens\": [";
    for (size_t i = 0; i < tokens.size(); i++) {
        m << (i ? ", " : "") << tokens[i];
    }
    m << "],\n  \"mode\": \"" << (st.light ? "light" : "all") << "\",\n  \"meta\": " << meta << ",\n  \"tensors\": [\n";
    for (size_t i = 0; i < st.entries.size(); i++) {
        m << st.entries[i] << (i + 1 < st.entries.size() ? ",\n" : "\n");
    }
    m << "  ]\n}\n";
    fprintf(stderr, "dumped %d tensors, %.1f MB\n", st.seq, st.bytes / 1e6);
    llama_backend_free();
    return 0;
}
