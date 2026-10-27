"""Doom eval: same weights, two decision mechanisms, identical controller code.

  wrapper   typed parallel judgements (goal / fire / target / item, then movement / dodge)
  baseline  one autoregressive JSON object per tick with the same fields; unparseable or
            out-of-schema output counts as a schema error and the tick holds position

Scored on game outcomes (kills, damage taken, deaths) and decision latency, over fixed
seeds so both agents face the same spawns.
"""

from __future__ import annotations

import json
import re
import time

from system_one.backends import DEFAULT_SYSTEM
from system_one.decider import Decider
from system_one.demos.doom import (
    DODGES,
    DOOM_PREAMBLE,
    MOVES,
    DoomEnv,
    feasible_goals,
    plan_text,
    state_text,
)
from system_one.types import Bool, Choice

ORDER = "Survive and kill as many enemies as possible. Pick up health when hurt."
FIRE_Q = "Is an enemy in the crosshair, so the player should fire this instant?"
GOAL_Q = "Considering the player, enemies and items, what is the player's highest-priority goal right now?"
TARGET_Q = "Which enemy should the player focus on?"
ITEM_Q = "Which item is most worth picking up now?"
MOVE_Q = "Given the current situation, how should the player move right now?"
DODGE_Q = "What does this exact moment call for to avoid damage?"


def questions_for(obs, enemy_opts, item_opts):
    return (
        [
            Choice("goal", GOAL_Q, options=feasible_goals(obs)),
            Bool("fire", FIRE_Q),
            Choice("target", TARGET_Q, options=enemy_opts),
            Choice("item", ITEM_Q, options=item_opts),
        ],
        [Choice("movement", MOVE_Q, options=MOVES), Choice("dodge", DODGE_Q, options=DODGES)],
    )


class WrapperAgent:
    name = "wrapper"

    def __init__(self, decider: Decider):
        self.d = decider

    def decide(self, obs, state, enemy_opts, item_opts) -> tuple[dict, int]:
        qa, qb = questions_for(obs, enemy_opts, item_opts)
        ra = self.d.decide(state, qa, preamble=DOOM_PREAMBLE.format(order=ORDER))
        goal = ra.answers["goal"]["choice"]
        target = ra.answers["target"]["choice"]
        item = ra.answers["item"]["choice"]
        plan = f"The player's current top priority is {goal.replace('_', ' ')}."
        rb = self.d.decide(state + "\n\n" + plan, qb, preamble=DOOM_PREAMBLE.format(order=ORDER))
        return {
            "goal": goal, "target": target, "item": item,
            "fire_p": ra.answers["fire"]["p_true"],
            "movement": rb.answers["movement"]["choice"], "dodge": rb.answers["dodge"]["choice"],
        }, 0


class BaselineAgent:
    name = "baseline"

    def __init__(self, decider: Decider, max_tokens: int = 120):
        self.be, self.max_tokens = decider.backend, max_tokens

    def decide(self, obs, state, enemy_opts, item_opts) -> tuple[dict, int]:
        goals = feasible_goals(obs)
        prompt = (
            f"{DOOM_PREAMBLE.format(order=ORDER)}\n\nSTATE:\n{state}\n\n"
            "Decide the player's action. Reply with ONLY one JSON object:\n"
            '{"goal": one of ' + json.dumps(goals) + ', "fire": true or false, '
            '"target": one of ' + json.dumps(enemy_opts) + ', "item": one of ' + json.dumps(item_opts) + ', '
            '"movement": one of ' + json.dumps(MOVES) + ', "dodge": one of ' + json.dumps(DODGES) + "}"
        )
        text = "".join(self.be.generate_stream(DEFAULT_SYSTEM, prompt, max_tokens=self.max_tokens))
        m = re.search(r"\{.*?\}", text, re.S)
        errors = 0
        obj = {}
        if m:
            try:
                obj = json.loads(m.group(0))
            except json.JSONDecodeError:
                errors += 1
        else:
            errors += 1

        def pick(key, allowed, default):
            nonlocal errors
            v = obj.get(key)
            if v in allowed:
                return v
            errors += 1
            return default

        fire = obj.get("fire")
        if not isinstance(fire, bool):
            errors += 1
            fire = False
        return {
            "goal": pick("goal", goals, goals[-1]),
            "target": pick("target", enemy_opts, "none"),
            "item": pick("item", item_opts, "none"),
            "fire_p": 1.0 if fire else 0.0,
            "movement": pick("movement", MOVES, "hold_ground"),
            "dodge": pick("dodge", DODGES, "no_dodge"),
        }, errors


def play(agent, seed: int, decisions: int, scenario: str = "defend_the_center.cfg", tics: int = 4,
         fire_threshold: float = 0.5, log=None) -> dict:
    env = DoomEnv(scenario, visible=False, realtime=False, tics=tics, seed=seed, record=None)
    latencies, errors, damage_taken = [], 0, 0.0
    last_health, deaths = None, 0
    kills = 0
    try:
        for step in range(decisions):
            env.respawn_if_dead()
            obs = env.observe()
            if obs is None:
                break
            if last_health is not None:
                if obs.health > last_health + 50:
                    deaths += 1
                elif obs.health < last_health:
                    damage_taken += last_health - obs.health
            last_health = obs.health
            kills = max(kills, int(obs.kills))
            state = state_text(obs)
            enemy_opts = ["none"] + [f"{t.tag} {t.name}" for t in obs.enemies]
            item_opts = ["none"] + [f"{t.tag} {t.name}" for t in obs.items]

            t0 = time.perf_counter()
            ans, errs = agent.decide(obs, state, enemy_opts, item_opts)
            latencies.append(time.perf_counter() - t0)
            errors += errs

            target = next((t for t in obs.enemies if f"{t.tag} {t.name}" == ans["target"]), None)
            item = next((t for t in obs.items if f"{t.tag} {t.name}" == ans["item"]), None)
            goal = ans["goal"]
            aim = target if goal == "attack_enemy" or (target and ans["fire_p"] >= fire_threshold) else None
            if goal.startswith("collect") and item:
                aim = item
            turn = max(-45.0, min(45.0, aim.bearing)) if aim is not None else (60.0 if obs.stuck else 6.0 if goal == "explore" else 0.0)
            move, dodge = ans["movement"], ans["dodge"]
            fwd = move == "advance" or (goal.startswith("collect") and item is not None) or goal == "explore"
            back = move == "back_off" or dodge == "dodge_back" or goal == "retreat"
            left = move == "strafe_left" or dodge == "dodge_left"
            right = move == "strafe_right" or dodge == "dodge_right"
            lined_up = target is not None and target.in_view and abs(target.bearing) < max(4.0, 400.0 / max(target.dist, 1.0))
            fire = bool(ans["fire_p"] >= fire_threshold and lined_up)
            env.act(fire, fwd and not back, back, left and not right, right and not left, obs.stuck, turn)
            if log:
                log.write(json.dumps({"agent": agent.name, "seed": seed, "step": step, "state": state,
                                      **ans, "fire": fire, "errors": errs, "latency_s": latencies[-1]}) + "\n")
    finally:
        kills = max(kills, int(env.g.get_game_variable(env.vzd.GameVariable.KILLCOUNT)))
        env.close()
    lat = sorted(latencies) or [0]
    return {
        "agent": agent.name, "seed": seed, "decisions": len(latencies), "kills": kills, "deaths": deaths,
        "damage_taken": damage_taken, "schema_errors": errors,
        "median_s": round(lat[len(lat) // 2], 3), "p90_s": round(lat[int(len(lat) * 0.9)], 3),
        "decisions_per_s": round(len(latencies) / max(sum(latencies), 1e-9), 2),
        "total_model_s": round(sum(latencies), 1),
    }


def run(agent, seeds: list[int], decisions: int, log=None, **kwargs) -> dict:
    episodes = [play(agent, seed, decisions, log=log, **kwargs) for seed in seeds]
    n = len(episodes)
    agg = {
        "agent": agent.name, "episodes": n, "decisions": sum(e["decisions"] for e in episodes),
        "kills": sum(e["kills"] for e in episodes), "deaths": sum(e["deaths"] for e in episodes),
        "damage_taken": sum(e["damage_taken"] for e in episodes),
        "schema_errors": sum(e["schema_errors"] for e in episodes),
        "kills_per_episode": round(sum(e["kills"] for e in episodes) / n, 2),
        "median_s": round(sum(e["median_s"] for e in episodes) / n, 3),
        "decisions_per_s": round(sum(e["decisions"] for e in episodes) / sum(e["total_model_s"] for e in episodes), 2),
        "per_episode": episodes,
    }
    return agg
