# usage: tokenizer_check.py <port> <model-name> <gguf>   compares loom-serve /tokenize with llama-tokenize on the same GGUF
import json, os, subprocess, sys, tempfile, urllib.request

port, model, gguf = sys.argv[1], sys.argv[2], sys.argv[3]
LLAMA = os.path.expanduser("~/inkling/inkB/build/bin")
PROMPTS = [
    "Hello, world!", "don't won't I'll we've They're", "The quick brown fox jumps over the lazy dog.",
    "  leading and trailing spaces  ", "line1\nline2\r\nline3\n\n\nline4", "tabs\t\tand    many     spaces",
    "1234567890 12 123 1234 3.14159 -42 1e10", "def f(x):\n    return x**2  # comment\n", "int main() { return 0; }",
    "CamelCaseIdentifier snake_case_name HTTPServer XMLHttpRequest", "ÄÖÜ äöü ß café naïve Straße",
    "日本語のテキスト 中文文本 한국어 텍스트", "emoji 😀🚀 mixed ​ zero-width", "https://example.com/path?q=1&r=2#frag",
    "<|message_user|><|content_text|>Hi<|end_message|><|content_model_end_sampling|>",
    "a" * 200, " " * 40 + "x", "\n" * 10, "MiXeD CaSe WoRdS ALLCAPS lowercase", "price: $12.50, 50% off!!! ...",
    "ΑΒΓ αβγ Привет мир", "x = [1, 2, 3]; y = {'a': 1}", "e.g. i.e. U.S.A. Ph.D.",
]
def llama(text):
    with tempfile.NamedTemporaryFile("wb", delete=False, suffix=".txt") as f:
        f.write(text.encode()); path = f.name
    out = subprocess.run([f"{LLAMA}/llama-tokenize", "-m", gguf, "--file", path, "--ids", "--no-bos", "--log-disable"],
                         capture_output=True, text=True, env={**os.environ, "LD_LIBRARY_PATH": LLAMA}).stdout.strip().splitlines()[-1]
    os.unlink(path)
    return json.loads(out)
def loom(text):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/tokenize", json.dumps({"model": model, "content": text, "add_special": False, "parse_special": True}).encode())
    return json.load(urllib.request.urlopen(req))["tokens"]
bad = 0
for p in PROMPTS:
    a, b = llama(p), loom(p)
    ok = a == b
    bad += not ok
    print("OK  " if ok else "DIFF", repr(p[:50]), len(a), len(b))
    if not ok:
        i = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        print("   first diff at", i, "llama", a[max(0, i-2):i+4], "loom", b[max(0, i-2):i+4])
print("mismatches:", bad, "of", len(PROMPTS))
