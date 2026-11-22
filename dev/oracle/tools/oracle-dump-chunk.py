#!/usr/bin/env python3
"""Patch oracle-dump.cpp (from oracle/tools) so that ORACLE_CHUNK=N decodes the prompt in chunks of N tokens.

Each chunk is dumped into <ORACLE_OUT>/c<i>/ with its own manifest.json (n_tokens, first position).
ORACLE_KEEP=regex (POSIX extended) limits the dump to tensors whose name matches.
usage: oracle-dump-chunk.py <path to examples/oracle-dump/oracle-dump.cpp>
"""
import sys

path = sys.argv[1]
src = open(path).read()

old = """    common_batch batch = common_batch_get_one(ctx, tokens);
    if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
        fprintf(stderr, "decode failed\\n");
        return 1;
    }
"""
new = """    const char * chunk_env = getenv("ORACLE_CHUNK");
    const size_t chunk = chunk_env ? (size_t) atol(chunk_env) : tokens.size();
    const std::string base_out = st.out;
    for (size_t c0 = 0, ci = 0; c0 < tokens.size(); c0 += chunk, ci++) {
        const size_t c1 = std::min(tokens.size(), c0 + chunk);
        std::vector<llama_token> part(tokens.begin() + c0, tokens.begin() + c1);
        if (chunk_env) {
            st.out = base_out + "/c" + std::to_string(ci);
            mkdir(st.out.c_str(), 0755);
            st.seq = 0;
            st.entries.clear();
            st.occ.clear();
        }
        common_batch batch = common_batch_get_one(ctx, part);
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
            fprintf(stderr, "decode failed\\n");
            return 1;
        }
        if (chunk_env) {
            std::ofstream mc(st.out + "/manifest.json");
            mc << "{\\n  \\"n_tokens\\": " << part.size() << ",\\n  \\"first\\": " << c0 << ",\\n  \\"tensors\\": [\\n";
            for (size_t i = 0; i < st.entries.size(); i++) {
                mc << st.entries[i] << (i + 1 < st.entries.size() ? ",\\n" : "\\n");
            }
            mc << "  ]\\n}\\n";
        }
    }
    st.out = base_out;
"""
assert old in src
src = src.replace(old, new)

old_w = """    const std::string n = t->name;
    if (n.find(" (") != std::string::npos) {
        return false;
    }
"""
new_w = """    const std::string n = t->name;
    if (n.find(" (") != std::string::npos) {
        return false;
    }
    if (const char * keep = getenv("ORACLE_KEEP")) {
        static std::regex re(keep);
        if (!std::regex_search(n, re)) {
            return false;
        }
    }
"""
assert old_w in src
src = src.replace(old_w, new_w)
src = src.replace("#include <map>\n", "#include <map>\n#include <regex>\n")
open(path, "w").write(src)
print("patched", path)
