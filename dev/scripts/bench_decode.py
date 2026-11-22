# usage: bench_decode.py <port> <model> [prompt]   per-token decode step cost from completion-token counts (nonstream), warmed
import http.client, json, sys, time, statistics

port, model = int(sys.argv[1]), sys.argv[2]
prompt = sys.argv[3] if len(sys.argv) > 3 else "Hello"
N = 40

def post(max_tokens):
    c = http.client.HTTPConnection("127.0.0.1", port)
    body = json.dumps({"model": model, "max_tokens": max_tokens, "messages": [{"role": "user", "content": prompt}]})
    t0 = time.perf_counter()
    c.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
    d = json.loads(c.getresponse().read())
    t = time.perf_counter() - t0
    c.close()
    return t, d["usage"]["completion_tokens"], d["usage"]["prompt_tokens"]

for _ in range(8): post(5)
rows = {}
for mt in (1, 2, 4, 6):
    ts = []
    for _ in range(N):
        t, n, p = post(mt); ts.append(t)
    rows[mt] = (statistics.median(ts) * 1e3, n, p)
    print(f"max_tokens={mt}: completion_tokens={n} prompt_tokens={p} median {rows[mt][0]:.2f} ms")
(a, na, _), (b, nb, _) = rows[1], rows[6]
print(f"decode step ~ {(b - a) / (nb - na):.3f} ms/token (slope from {na} to {nb} tokens), request floor {a:.2f} ms")
