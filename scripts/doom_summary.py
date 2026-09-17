"""Summarize a Doom decision log."""
import collections
import json
import sys

rows = [json.loads(l) for l in open(sys.argv[1])]
lat = sorted(r["latency_s"] for r in rows)
print(f"{len(rows)} decisions  median {lat[len(lat)//2]*1000:.0f} ms  p90 {lat[int(len(lat)*.9)]*1000:.0f} ms")
print("goals ", dict(collections.Counter(r["answers"]["goal"]["choice"] for r in rows)))
print("moves ", dict(collections.Counter(r["answers"]["movement"]["choice"] for r in rows)))
print("fired ", sum(r["fire"] for r in rows), " with enemy in view:", sum("(not in view)" not in r["state"] and "known enemies:" not in r["state"] for r in rows))
kills = [int(r["state"].split("kills ")[1].split()[0]) for r in rows]
hp = [int(r["state"].split("health ")[1].split()[0]) for r in rows]
print("kills ", max(kills), " min health", min(hp), " deaths", sum(1 for a, b in zip(hp, hp[1:]) if b > a + 50))
