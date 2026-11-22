import http.client, json, sys, time, statistics, concurrent.futures

port = int(sys.argv[1]); model = sys.argv[2]; prompt = sys.argv[3] if len(sys.argv) > 3 else "abxa"
N = 60

def post(max_tokens, stream=False):
    c = http.client.HTTPConnection("127.0.0.1", port)
    body = json.dumps({"model": model, "max_tokens": max_tokens, "stream": stream,
                       "messages": [{"role": "user", "content": prompt}]})
    t0 = time.perf_counter()
    c.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
    r = c.getresponse()
    stamps = []
    if stream:
        for line in r:
            if line.startswith(b"data: {"):
                stamps.append(time.perf_counter() - t0)
        data = None
    else:
        data = json.loads(r.read())
    t = time.perf_counter() - t0
    c.close()
    return t, stamps, data

def med(xs): return statistics.median(xs) * 1e3

for _ in range(10): post(8)
res = {}
for mt in (1, 4, 8):
    ts = [post(mt)[0] for _ in range(N)]
    res[mt] = med(ts)
    print(f"nonstream max_tokens={mt}: median {med(ts):.2f} ms  min {min(ts)*1e3:.2f}  p90 {sorted(ts)[int(N*.9)]*1e3:.2f}")
print(f"per-token slope (mt8-mt1)/7: {(res[8]-res[1])/7:.3f} ms/token; fixed floor (mt1): {res[1]:.2f} ms")
gaps, first = [], []
for _ in range(N):
    t, s, _ = post(8, True)
    first.append(s[1]); gaps += [b - a for a, b in zip(s[1:-1], s[2:-1])]
print(f"stream: first content chunk median {med(first):.2f} ms; inter-chunk gap median {med(gaps):.3f} ms p90 {sorted(gaps)[int(len(gaps)*.9)]*1e3:.3f}")
for conc in (4, 8):
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(conc) as ex:
        out = list(ex.map(lambda _: post(8)[0], range(N)))
    wall = time.perf_counter() - t0
    print(f"concurrency {conc}: {N} req in {wall*1e3:.0f} ms = {N/wall:.0f} req/s, {N*8/wall:.0f} tok/s, per-req median {med(out):.2f} ms")
