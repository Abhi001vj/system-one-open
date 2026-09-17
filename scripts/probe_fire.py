"""Probe how question phrasing changes the fire decision in a clear-shot Doom state."""
import sys
from system_one import Bool, Decider

STATE = """health 100 | armor 0 | weapon pistol with 50 ammo | kills 0
known enemies (in view first, nearest first):
  E1 MarineChainsawVzd: distance 373, dead ahead (in crosshair)
visible items (nearest first):
  I3 GreenArmor: distance 841, 7° right"""
EMPTY = """health 100 | armor 0 | weapon pistol with 50 ammo | kills 0
known enemies: none
visible items: none"""
PRE = "You are the tactical brain of a Doom player in a video game. Standing orders: Survive and kill as many enemies as possible."
QS = [
    Bool("trigger", "Should the player's trigger be held down right now?"),
    Bool("shoot", "Should the player shoot right now?"),
    Bool("crosshair", "Is an enemy in the crosshair, so the player should fire this instant?"),
    Bool("clear_shot", "Does the player have a clear shot at an enemy right now?"),
]
d = Decider(sys.argv[1])
for name, st in [("enemy ahead", STATE), ("no enemies", EMPTY)]:
    r = d.decide(st, QS, preamble=PRE)
    print(f"{name:12s}", {k: v["p_true"] for k, v in r.answers.items()}, f"{r.latency_s*1000:.0f} ms")
