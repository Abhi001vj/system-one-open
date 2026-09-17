import json, sys, time
from system_one import Decider
from tests.test_equivalence import STATE, QUESTIONS
d = Decider(sys.argv[1])
for i in range(3):
    r = d.decide(STATE, QUESTIONS)
    print(r.backend, f"{r.latency_s*1000:.0f}ms", r.stats)
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'legend'} for k, v in r.answers.items()}))
