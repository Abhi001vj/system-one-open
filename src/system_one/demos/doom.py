"""Doom agent: structured game state → typed judgements → code turns them into buttons.

Like the TypeSafe demo, the model reads game state as text (not pixels) and code
executes. Each decision cycle is two parallel scoring passes:

  stage A  goal (choice) · target enemy (choice) · target item (choice) · fire (bool)
  stage B  movement (choice) · dodge (choice)      — conditioned on stage A's plan

Aiming is a deterministic controller toward the chosen target's bearing; the
trigger is pulled only when the model wants to fire AND the target is lined up.

    s1-doom --model qwen2.5-1.5b
    s1-doom --model llamacpp --order "Prioritise shotgun guys. Collect health below 50%."
    s1-doom --realtime          # game keeps running while the model thinks
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, field

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..decider import Decider
from ..types import Bool, Choice

MONSTERS = {
    "Zombieman", "ShotgunGuy", "ChaingunGuy", "DoomImp", "Demon", "Spectre", "LostSoul", "Cacodemon",
    "HellKnight", "BaronOfHell", "Arachnotron", "PainElemental", "Revenant", "Fatso", "Archvile",
    "SpiderMastermind", "Cyberdemon", "WolfensteinSS", "MarineChainsawVzd",
}
HEALTH = {"Medikit", "Stimpack", "HealthBonus", "Soulsphere", "Megasphere", "Berserk"}
AMMO_WEAPONS = {
    "Clip", "ClipBox", "Shell", "ShellBox", "RocketAmmo", "RocketBox", "Cell", "CellPack", "Backpack",
    "Shotgun", "SuperShotgun", "Chaingun", "RocketLauncher", "PlasmaRifle", "BFG9000", "Chainsaw",
}
ARMOR = {"GreenArmor", "BlueArmor", "ArmorBonus"}
WEAPON_NAMES = {1: "fist/chainsaw", 2: "pistol", 3: "shotgun", 4: "chaingun", 5: "rocket launcher", 6: "plasma rifle", 7: "BFG"}

MOVES = ["hold_ground", "advance", "back_off", "strafe_left", "strafe_right"]
DODGES = ["no_dodge", "dodge_left", "dodge_right", "dodge_back"]


@dataclass
class Thing:
    tag: str
    name: str
    dist: float
    bearing: float  # degrees, + = left of crosshair
    in_view: bool = True


@dataclass
class Obs:
    tick: int
    health: float
    armor: float
    ammo: float
    weapon: int
    kills: int
    x: float
    y: float
    angle: float
    enemies: list[Thing] = field(default_factory=list)
    items: list[Thing] = field(default_factory=list)
    damage_recent: float = 0.0
    stuck: bool = False


def wrap180(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


class DoomEnv:
    def __init__(self, scenario: str, visible: bool, realtime: bool, tics: int, seed: int | None, record: str | None):
        import vizdoom as vzd

        self.vzd = vzd
        g = vzd.DoomGame()
        g.load_config(os.path.join(vzd.scenarios_path, scenario))
        g.set_window_visible(visible)
        if visible:
            g.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
        g.set_objects_info_enabled(True)
        g.set_labels_buffer_enabled(True)
        g.set_render_hud(True)
        g.clear_available_buttons()
        self.buttons = [
            vzd.Button.ATTACK, vzd.Button.MOVE_FORWARD, vzd.Button.MOVE_BACKWARD,
            vzd.Button.MOVE_LEFT, vzd.Button.MOVE_RIGHT, vzd.Button.USE, vzd.Button.TURN_LEFT_RIGHT_DELTA,
        ]
        for b in self.buttons:
            g.add_available_button(b)
        g.clear_available_game_variables()
        V = vzd.GameVariable
        for v in [V.HEALTH, V.ARMOR, V.SELECTED_WEAPON_AMMO, V.SELECTED_WEAPON, V.KILLCOUNT, V.POSITION_X, V.POSITION_Y, V.ANGLE]:
            g.add_available_game_variable(v)
        g.set_episode_timeout(0)
        g.set_mode(vzd.Mode.ASYNC_PLAYER if realtime else vzd.Mode.PLAYER)
        if seed is not None:
            g.set_seed(seed)
        g.init()
        g.new_episode(record or "")
        self.g, self.tics = g, tics
        self.health_hist: list[float] = []
        self.last_xy = None
        self.dead: set[int] = set()
        self.hearing_range = 600.0

    def observe(self) -> Obs | None:
        if self.g.is_episode_finished():
            return None
        s = self.g.get_state()
        if s is None:
            return None
        hp, armor, ammo, weapon, kills, x, y, angle = s.game_variables
        visible = {}
        for lab in s.labels:
            # flattened sprites are corpses; ignore them
            if lab.object_name in MONSTERS and lab.height < 0.7 * lab.width:
                continue
            visible[lab.object_id] = lab
        enemies, items = [], []
        for o in s.objects:
            if o.name == "DoomPlayer":
                continue
            in_view = o.id in visible
            dx, dy = o.position_x - x, o.position_y - y
            dist = math.hypot(dx, dy)
            bearing = wrap180(math.degrees(math.atan2(dy, dx)) - angle)
            thing = Thing("", o.name, dist, bearing, in_view)
            # enemies out of view still matter when close (they are shooting you)
            if o.name in MONSTERS and (in_view or dist < self.hearing_range) and o.id not in self.dead:
                enemies.append(thing)
            elif o.name in HEALTH | AMMO_WEAPONS | ARMOR and in_view:
                items.append(thing)
        for lab in s.labels:  # remember corpses so out-of-view enemies exclude them
            if lab.object_name in MONSTERS and lab.height < 0.7 * lab.width:
                self.dead.add(lab.object_id)
        enemies.sort(key=lambda t: (not t.in_view, t.dist))
        items.sort(key=lambda t: t.dist)
        enemies, items = enemies[:6], items[:6]
        for i, t in enumerate(enemies):
            t.tag = f"E{i + 1}"
        for i, t in enumerate(items):
            t.tag = f"I{i + 1}"
        self.health_hist = (self.health_hist + [hp])[-9:]  # ~1 s at 4 tics/decision
        stuck = self.last_xy is not None and math.hypot(x - self.last_xy[0], y - self.last_xy[1]) < 2.0
        self.last_xy = (x, y)
        return Obs(
            s.tic, hp, armor, ammo, int(weapon), int(kills), x, y, angle, enemies, items,
            damage_recent=max(0.0, self.health_hist[0] - hp), stuck=stuck,
        )

    def act(self, fire: bool, fwd: bool, back: bool, left: bool, right: bool, use: bool, turn_deg: float) -> float:
        # TURN_LEFT_RIGHT_DELTA is applied per tic; positive turns right
        per_tic = -turn_deg / self.tics
        return self.g.make_action([float(fire), float(fwd), float(back), float(left), float(right), float(use), per_tic], self.tics)

    def respawn_if_dead(self):
        if self.g.is_player_dead():
            self.g.respawn_player()
            self.health_hist.clear()

    def close(self):
        self.g.close()


def describe_dir(b: float) -> str:
    if abs(b) < 4:
        return "dead ahead (in crosshair)"
    if abs(b) > 135:
        return "behind the player"
    return f"{abs(b):.0f}° {'left' if b > 0 else 'right'}"


def state_text(o: Obs) -> str:
    lines = [
        f"health {o.health:.0f}" + (f" (took {o.damage_recent:.0f} damage in the last second)" if o.damage_recent else "")
        + f" | armor {o.armor:.0f} | weapon {WEAPON_NAMES.get(o.weapon, o.weapon)} with {o.ammo:.0f} ammo | kills {o.kills}"
        + (" | player appears stuck against a wall" if o.stuck else ""),
        "known enemies (in view first, nearest first):" if o.enemies else "known enemies: none",
    ]
    lines += [
        f"  {t.tag} {t.name}: distance {t.dist:.0f}, {describe_dir(t.bearing)}{'' if t.in_view else ' (not in view)'}"
        for t in o.enemies
    ]
    lines.append("visible items (nearest first):" if o.items else "visible items: none")
    lines += [f"  {t.tag} {t.name}: distance {t.dist:.0f}, {describe_dir(t.bearing)}" for t in o.items]
    return "\n".join(lines)


def feasible_goals(o: Obs) -> list[str]:
    """Code decides what is possible; the model only ranks possibilities."""
    names = {t.name for t in o.items}
    goals = []
    if o.enemies:
        goals += ["attack_enemy", "retreat"]
    if names & HEALTH:
        goals.append("collect_health")
    if names & AMMO_WEAPONS:
        goals.append("collect_ammo_or_weapon")
    if names & ARMOR:
        goals.append("collect_armor")
    return goals + ["explore"]


def plan_text(goal: str, target: Thing | None, item: Thing | None) -> str:
    focus = f", focusing on {target.tag} {target.name}" if goal in ("attack_enemy", "retreat") and target else ""
    if goal.startswith("collect") and item:
        focus = f", heading for {item.tag} {item.name}"
    return f"The player's current top priority is {goal.replace('_', ' ')}{focus}."


def bar(p: float, width: int = 22) -> str:
    n = round(p * width)
    return "█" * n + "·" * (width - n)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen2.5-1.5b")
    ap.add_argument("--scenario", default="deathmatch.cfg", help="any ViZDoom scenario cfg (deathmatch, defend_the_center, deadly_corridor…)")
    ap.add_argument("--order", default="Survive and kill as many enemies as possible. Pick up health when hurt.", help="standing orders in plain English")
    ap.add_argument("--tics", type=int, default=4, help="game tics per decision (35 tics = 1 s)")
    ap.add_argument("--decisions", type=int, default=600)
    ap.add_argument("--fire-threshold", type=float, default=0.5)
    ap.add_argument("--realtime", action="store_true", help="async mode: the game does not wait for the model")
    ap.add_argument("--single-stage", action="store_true", help="one pass for all questions (faster, no plan conditioning)")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--log", default=None, help="JSONL of every state/decision (default runs/doom-<time>.jsonl)")
    ap.add_argument("--record", default=None, help="ViZDoom .lmp demo file to record the episode")
    args = ap.parse_args()

    console = Console()
    with console.status(f"loading {args.model}…"):
        decider = Decider(args.model)
    env = DoomEnv(args.scenario, not args.headless, args.realtime, args.tics, args.seed, args.record)
    os.makedirs("runs", exist_ok=True)
    log_path = args.log or f"runs/doom-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    log = open(log_path, "w")
    preamble = f"You are the tactical brain of a Doom player. Standing orders: {args.order}"
    latencies: list[float] = []
    panel = {"a": None, "b": None, "plan": "", "obs": "", "action": ""}

    def render():
        tbl = Table.grid(padding=(0, 1))
        tbl.add_column(style="bold green", no_wrap=True)
        tbl.add_column(no_wrap=True)
        tbl.add_column(style="dim", no_wrap=True)
        for res in (panel["a"], panel["b"]):
            if not res:
                continue
            for key, a in res.answers.items():
                if a["type"] == "bool":
                    rows = [("true", a["p_true"]), ("false", 1 - a["p_true"])]
                else:
                    rows = sorted(a["probabilities"].items(), key=lambda kv: -kv[1])[:4]
                tbl.add_row(key.upper(), "", "")
                for i, (name, p) in enumerate(rows):
                    tbl.add_row("", Text(bar(p), style="green" if i == 0 else "dim green"), f"{p:.2f} {name}")
        avg = sum(latencies[-20:]) / max(len(latencies[-20:]), 1)
        head = Text(f"{decider.backend.describe()}  ·  last {latencies[-1] * 1000 if latencies else 0:.0f} ms  ·  avg {avg * 1000:.0f} ms  ·  {1 / avg if avg else 0:.1f} decisions/s", style="bold")
        return Panel(
            Group(head, Text(panel["plan"], style="yellow"), Text(panel["action"], style="cyan"), Text(""), tbl, Text(""), Text(panel["obs"], style="dim")),
            title="DOOM // judgements", border_style="green",
        )

    try:
        with Live(render(), console=console, refresh_per_second=10) as live:
            for step in range(args.decisions):
                env.respawn_if_dead()
                o = env.observe()
                if o is None:
                    break
                st = state_text(o)
                enemy_opts = ["none"] + [f"{t.tag} {t.name}" for t in o.enemies]
                item_opts = ["none"] + [f"{t.tag} {t.name}" for t in o.items]
                goals = feasible_goals(o)
                qa = [
                    Choice("goal", "Considering the player, enemies and items, what is the player's highest-priority goal right now?", options=goals),
                    Bool("fire", "Is an enemy in the crosshair, so the player should fire this instant?"),
                    # with nothing in view these have one option and are resolved without the model
                    Choice("target", "Which enemy should the player focus on?", options=enemy_opts),
                    Choice("item", "Which item is most worth picking up now?", options=item_opts),
                ]
                qb = [
                    Choice("movement", "Given the current situation, how should the player move right now?", options=MOVES),
                    Choice("dodge", "What does this exact moment call for to avoid damage?", options=DODGES),
                ]
                t0 = time.perf_counter()
                rb = None
                if args.single_stage:
                    ra = decider.decide(st, qa + qb, preamble=preamble)
                else:
                    ra = decider.decide(st, qa, preamble=preamble)
                ans = ra.answers
                goal, tgt_name, item_name = ans["goal"]["choice"], ans["target"]["choice"], ans["item"]["choice"]
                target = next((t for t in o.enemies if f"{t.tag} {t.name}" == tgt_name), None)
                item = next((t for t in o.items if f"{t.tag} {t.name}" == item_name), None)
                plan = plan_text(goal, target, item)
                if not args.single_stage:
                    rb = decider.decide(st + "\n\n" + plan, qb, preamble=preamble)
                    ans = {**ra.answers, **rb.answers}
                latencies.append(time.perf_counter() - t0)

                # ---- code: judgements -> buttons
                aim = target if goal in ("attack_enemy",) or (target and ans["fire"]["p_true"] >= args.fire_threshold) else None
                if goal.startswith("collect") and item:
                    aim = item
                turn = 0.0
                if aim is not None:
                    turn = max(-45.0, min(45.0, aim.bearing))
                elif o.stuck:
                    turn = 60.0
                elif goal == "explore":
                    turn = 6.0  # slow sweep to find things
                move, dodge = ans["movement"]["choice"], ans["dodge"]["choice"]
                fwd = move == "advance" or (goal.startswith("collect") and item is not None) or (goal == "explore")
                back = move == "back_off" or dodge == "dodge_back" or goal == "retreat"
                left = move == "strafe_left" or dodge == "dodge_left"
                right = move == "strafe_right" or dodge == "dodge_right"
                lined_up = target is not None and abs(target.bearing) < max(4.0, 400.0 / max(target.dist, 1.0))
                fire = bool(ans["fire"]["p_true"] >= args.fire_threshold and lined_up)
                env.act(fire, fwd and not back, back, left and not right, right and not left, o.stuck, turn)

                panel.update(
                    a=ra, b=rb, plan=plan, obs=st,
                    action=f"buttons: {'FIRE ' if fire else ''}{'FWD ' if fwd and not back else ''}{'BACK ' if back else ''}"
                    f"{'LEFT ' if left and not right else ''}{'RIGHT ' if right and not left else ''}turn {turn:+.0f}°",
                )
                live.update(render())
                log.write(json.dumps({"step": step, "tick": o.tick, "state": st, "plan": plan, "answers": ans, "latency_s": latencies[-1], "fire": fire, "turn": turn}) + "\n")
    except KeyboardInterrupt:
        pass
    finally:
        kills = env.g.get_game_variable(env.vzd.GameVariable.KILLCOUNT)
        env.close()
        log.close()
    if latencies:
        lat = sorted(latencies)
        console.print(
            f"decisions {len(lat)}  ·  median {lat[len(lat) // 2] * 1000:.0f} ms  ·  p90 {lat[int(len(lat) * 0.9)] * 1000:.0f} ms"
            f"  ·  kills {kills:.0f}  ·  log {log_path}"
        )


if __name__ == "__main__":
    main()
