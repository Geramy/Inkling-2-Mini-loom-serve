"""Reader for HAL profile bundles (.irpf, file format 2.0). usage: irpf.py <bundle> [summary]
Struct layouts follow runtime/src/iree/hal/profile_schema.h (hrx-system v0.2.0); every record carries its own length, a size mismatch is reported."""
import struct
import sys
from collections import Counter, defaultdict

REC_HDR = struct.Struct("<7Q5I4xQQIHH")
assert REC_HDR.size == 104, REC_HDR.size
CT = "application/vnd.iree.hal.profile."
DISPATCH = struct.Struct("<IIQQQQIIIIIIIIQQ")
QUEUE_EV = struct.Struct("<IIIIQqQQQQIIIIIIQq")
DEVICE = struct.Struct("<IIII16sQ")
FUNC = struct.Struct("<IIQIIIIIIIIQQ")
CLOCK = struct.Struct("<IIIIQQQQQqq")
EXEC = struct.Struct("<IIQ")


def records(path):
    data = open(path, "rb").read()
    magic, vmaj, vmin, hlen, flags, flen = struct.unpack_from("<IHHIIQ", data, 0)
    assert magic == 0x46505249, hex(magic)
    end = flen or len(data)
    off = hlen
    while off < end:
        f = REC_HDR.unpack_from(data, off)
        rl, pl = f[0], f[1]
        hl, ctl, nl = f[7], f[8], f[9]
        rtype = f[15]
        p = off + hl
        ctype = data[p:p + ctl].decode()
        name = data[p + ctl:p + ctl + nl].decode(errors="replace")
        payload = data[p + ctl + nl:p + ctl + nl + pl]
        yield {"type": rtype, "ctype": ctype.replace(CT, ""), "name": name, "payload": payload, "stream": f[3], "dropped": f[13],
               "queue": f[11], "dev": f[10], "version": (vmaj, vmin)}
        off += rl


def arrays(payload, st, label):
    off = 0
    out = []
    n = len(payload)
    while off + 4 <= n:
        rl = struct.unpack_from("<I", payload, off)[0]
        if rl < 4 or off + rl > n:
            break
        if rl < st.size:
            raise SystemExit(f"{label}: record length {rl} < struct {st.size}")
        out.append((st.unpack_from(payload, off), payload[off + st.size:off + rl]))
        off += rl
    return out


def load(path):
    b = {"freq": {}, "funcs": {}, "disp": [], "queue": [], "cb": set(), "dropped": 0, "ctypes": Counter(), "clock": [], "version": None}
    for r in records(path):
        b["version"] = r["version"]
        ct = r["ctype"]
        b["ctypes"][ct] += 1
        b["dropped"] += r["dropped"]
        p = r["payload"]
        if ct == "devices":
            for f, _ in arrays(p, DEVICE, ct):
                if f[1] & 2:
                    b["freq"][f[2]] = f[5]
        elif ct == "executable-functions":
            for f, nm in arrays(p, FUNC, ct):
                b["funcs"][(f[2], f[3])] = nm[:f[10]].decode(errors="replace")
        elif ct == "dispatch-events":
            for f, _ in arrays(p, DISPATCH, ct):
                b["disp"].append((f[2], f[3], f[4], f[5], f[6], f[7], f[8:11], f[11:14], f[14], f[15], f[1], r["dev"]))
        elif ct == "queue-events":
            for f, _ in arrays(p, QUEUE_EV, ct):
                b["queue"].append(f)
        elif ct == "clock-correlations":
            for f, _ in arrays(p, CLOCK, ct):
                b["clock"].append(f)
    return b


if __name__ == "__main__":
    b = load(sys.argv[1])
    print("version", b["version"], "dropped", b["dropped"], "freq", b["freq"])
    print("record types", dict(b["ctypes"]))
    print("functions", len(b["funcs"]), "dispatch events", len(b["disp"]), "queue events", len(b["queue"]))
    d = b["disp"]
    print("flags", Counter(x[10] for x in d))
    print("command buffers", len({x[2] for x in d}), "submissions", len({x[1] for x in d}))
    per = Counter(x[1] for x in d)
    print("dispatches per submission (top)", Counter(per.values()).most_common(8))
    print("queue types", Counter(q[1] for q in b["queue"]))
