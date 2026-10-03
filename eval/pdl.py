#!/usr/bin/env python3
"""Parallel ranged download of one file, then sha256 check.  usage: pdl.py <url> <out> <expected_sha256> [threads]"""
import hashlib
import os
import sys
import threading
import time
import urllib.request

url, out, want = sys.argv[1], sys.argv[2], sys.argv[3]
nthreads = int(sys.argv[4]) if len(sys.argv) > 4 else 8
CHUNK = 64 * 1024 * 1024

with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as r:   # follows the redirect
    final, size = r.geturl(), int(r.headers["Content-Length"])
fd = os.open(out, os.O_RDWR | os.O_CREAT, 0o644)
os.ftruncate(fd, size)
chunks = [(o, min(o + CHUNK, size) - 1) for o in range(0, size, CHUNK)]
lock, done = threading.Lock(), [0]
t0 = time.time()


def worker():
    while True:
        with lock:
            if not chunks:
                return
            a, b = chunks.pop(0)
        for attempt in range(8):
            try:
                with urllib.request.urlopen(urllib.request.Request(final, headers={"Range": f"bytes={a}-{b}"}), timeout=60) as r:
                    data = r.read()
                if len(data) != b - a + 1:
                    raise IOError(f"short read {len(data)} for {a}-{b}")
                os.pwrite(fd, data, a)
                break
            except Exception:
                time.sleep(2 + attempt * 2)
                if attempt == 7:
                    raise
        with lock:
            done[0] += b - a + 1


threads = [threading.Thread(target=worker) for _ in range(nthreads)]
for t in threads:
    t.start()
while any(t.is_alive() for t in threads):
    time.sleep(10)
    print(f"{done[0] / 1e9:.1f}/{size / 1e9:.1f} GB  {done[0] / 1e6 / (time.time() - t0):.0f} MB/s", flush=True)
os.close(fd)
h = hashlib.sha256()
with open(out, "rb") as f:
    for blk in iter(lambda: f.read(16 * 1024 * 1024), b""):
        h.update(blk)
got = h.hexdigest()
print(("SHA256 OK " if got == want else "SHA256 MISMATCH ") + got, flush=True)
sys.exit(0 if got == want else 1)
