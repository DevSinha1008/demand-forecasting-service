"""Load test the running service -> produces CV slots [H]/[I]/[J]/[K].

Run against docker compose stack:
    python scripts/loadtest.py http://localhost:8000

Method: measure two phases:
  cold  — unique (uncached) requests only  -> uncached p99
  hot   — repeated horizons (cache hits)   -> cached p99, hit rate
[J] = % reduction of cached p99 vs uncached p99. Report req/sec [I] and
p99 [H] from the hot phase (the realistic serving mix).
"""
import sys, time, concurrent.futures as cf
import httpx

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
N_COLD, N_HOT, WORKERS = 60, 2000, 16

def timed_get(client, url):
    t0 = time.perf_counter()
    r = client.get(url, timeout=30)
    dt = (time.perf_counter() - t0) * 1000
    return dt, r.json().get("cache")

def pct(v, p):
    s = sorted(v); return s[int(p * (len(s) - 1))]

with httpx.Client() as c:
    c.get(f"{BASE}/health").raise_for_status()
    cold = []
    for h in range(1, N_COLD + 1):        # unique horizons => cache misses
        dt, tag = timed_get(c, f"{BASE}/forecast?h={h}")
        if tag == "miss": cold.append(dt)

hot, hits = [], 0
def one(i):
    with httpx.Client() as cc:
        return timed_get(cc, f"{BASE}/forecast?h={(i % 24) + 1}")
t0 = time.perf_counter()
with cf.ThreadPoolExecutor(WORKERS) as ex:
    for dt, tag in ex.map(one, range(N_HOT)):
        hot.append(dt); hits += (tag == "hit")
wall = time.perf_counter() - t0

print(f"uncached p99        : {pct(cold, 0.99):8.1f} ms   (n={len(cold)})")
print(f"cached-mix p99  [H] : {pct(hot, 0.99):8.1f} ms   (n={len(hot)})")
print(f"throughput      [I] : {len(hot)/wall:8.1f} req/sec ({WORKERS} workers)")
print(f"cache hit rate  [K] : {100*hits/len(hot):8.1f} %")
print(f"p99 reduction   [J] : {100*(pct(cold,0.99)-pct(hot,0.99))/pct(cold,0.99):8.1f} %")
