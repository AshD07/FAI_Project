"""
Browser bridge for the live web demo.

Runs the project's own simulation and AutonomousPilot (unchanged code from src/)
one tick at a time and returns a JSON snapshot for the canvas renderer.
Rendering/export-only libraries (pygame, pandas, matplotlib) are replaced by
inert stubs because the browser draws the scene itself; none of them is used
by the decision-making code path.
"""
import json
import sys
import types


class _Dummy:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return _Dummy()

    def __getattr__(self, name):
        return _Dummy()


class _StubModule(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Dummy


for _name in ("pygame", "pandas", "matplotlib", "matplotlib.pyplot"):
    if sys.modules.get(_name) is None:
        try:
            __import__(_name)
        except Exception:
            sys.modules[_name] = _StubModule(_name)

from src.agent.autonomous_pilot import AutonomousPilot  # noqa: E402
from src.config import GameConfig  # noqa: E402
from src.environment.game_engine import GameEngine  # noqa: E402
from src.optimization.genetic_algorithm import AgentGenome  # noqa: E402
from src.sensors.radar_sensor import RadarSensor  # noqa: E402

SCENARIO_NAMES = {
    "s1": "1_Target_Accuracy",
    "s2": "2_Hazard_Swarm",
    "s3": "3_Passive_Rival",
    "s4": "4_Adversarial_Duel",
    "s5": "5_High_Speed_Debris",
}


def _load_ga_genome(path="best_optimized_genome.json"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return AgentGenome(**json.load(f)["parameters"])
    except Exception:
        return None


def _r(x, nd=1):
    return round(float(x), nd)


class LiveDemo:
    """One episode: engine + radar + pilot (+ scripted rival in rival scenarios)."""

    def __init__(self, seed=1, mode="solo", params="ga", time_limit=60.0):
        self.seed = int(seed)
        self.mode = mode
        self.params = params
        cfg = GameConfig()
        self.w, self.h = cfg.arena.width, cfg.arena.height
        self.rival_bot = None

        if mode == "solo":
            # Same setup as run_survival.py: one life, continuous asteroid replenishment
            self.engine = GameEngine(config=cfg, seed=self.seed, auto_replenish=True)
            self.sensor = RadarSensor(config=cfg.sensor, seed=self.seed + 100)
        else:
            # Same setup as the benchmark evaluator
            from src.evaluation.benchmark_suite import BENCHMARK_SCENARIOS, setup_scenario_engine
            scen = next(s for s in BENCHMARK_SCENARIOS if s.name == SCENARIO_NAMES[mode])
            self.engine, self.sensor, self.rival_bot = setup_scenario_engine(scen, seed=self.seed)
        self.engine.time_limit = float(time_limit)

        genome = _load_ga_genome() if params == "ga" else None
        self.pilot = AutonomousPilot(arena_width=self.w, arena_height=self.h, genome=genome)

        # Capture the tracker's output (Kalman estimates) for drawing, without changing behaviour
        self._tracks = []
        original_update = self.pilot.tracker.update

        def capture(obs):
            result = original_update(obs)
            self._tracks = result
            return result

        self.pilot.tracker.update = capture

        self.tick = 0
        self.rule_counts = {}
        self.last = None
        self.last_obs = None
        self.plan_origin = None

    # ------------------------------------------------------------------
    def step(self, n=1):
        for _ in range(int(n)):
            if self.engine.is_game_over():
                break
            obs = self.sensor.observe(self.engine, observing_ship_id=1)
            turn, thrust, shoot, mine, dec = self.pilot.decide(obs)
            self.last, self.last_obs = dec, obs
            self.plan_origin = (float(obs.telemetry.position[0]), float(obs.telemetry.position[1]))
            self.rule_counts[dec.active_rule] = self.rule_counts.get(dec.active_rule, 0) + 1

            actions = {1: (turn, thrust, shoot, mine)}
            if self.rival_bot and 2 in self.engine.ships and self.engine.ships[2].is_alive:
                actions[2] = self.rival_bot.decide(self.engine.ships[2].to_dict(), self.engine.get_true_state())
            elif 2 in self.engine.ships:
                actions[2] = (0.0, 0.0, False, False)
            self.engine.step(actions)
            self.tick += 1
        return json.dumps(self.snapshot())

    # ------------------------------------------------------------------
    def snapshot(self):
        eng = self.engine
        me = eng.ships[1]
        ships = []
        for sid, s in eng.ships.items():
            ships.append({
                "id": sid, "x": _r(s.position[0]), "y": _r(s.position[1]),
                "heading": _r(s.heading), "r": _r(s.radius), "alive": bool(s.is_alive),
                "inv": bool(getattr(s, "invulnerable_timer", 0.0) > 0.0),
            })
        state = {
            "t": _r(eng.time, 2),
            "tick": self.tick,
            "over": bool(eng.is_game_over()),
            "alive": bool(me.is_alive),
            "score": int(me.score),
            "shots": int(me.shots_fired),
            "hits": int(me.shots_hit),
            "acc": _r(me.accuracy * 100.0, 1),
            "ammo": int(me.ammo),
            "speed": _r((me.velocity[0] ** 2 + me.velocity[1] ** 2) ** 0.5),
            "ships": ships,
            "asteroids": [
                {"id": int(a.id), "x": _r(a.position[0]), "y": _r(a.position[1]), "r": _r(a.radius), "size": int(a.size)}
                for a in eng.asteroids if a.is_alive
            ],
            "bullets": [
                {"x": _r(b.position[0]), "y": _r(b.position[1]), "own": int(b.owner_id)}
                for b in eng.bullets if b.is_alive
            ],
            "explosions": [
                {"x": _r(e.position[0]), "y": _r(e.position[1]), "r": _r(e.current_radius), "a": _r(1.0 - e.time_alive / e.lifetime, 2)}
                for e in getattr(eng, "explosions", []) if e.is_alive
            ],
            "rules": self.rule_counts,
        }

        if self.last_obs is not None:
            state["radar"] = [
                [_r(d.measured_position[0]), _r(d.measured_position[1])]
                for d in list(self.last_obs.radar_asteroids) + list(self.last_obs.radar_rivals)
            ]
        state["tracks"] = [
            {
                "id": int(t.track_id), "x": _r(t.position[0]), "y": _r(t.position[1]),
                "vx": _r(t.velocity[0]), "vy": _r(t.velocity[1]), "ok": bool(t.is_confirmed),
                "type": t.entity_type,
                "traj": [[_r(p[0]), _r(p[1])] for p in (t.future_trajectory or [])[4::5]],
            }
            for t in self._tracks
        ]

        dec = self.last
        if dec is not None:
            d = {
                "rule": dec.active_rule,
                "why": dec.active_explanation,
                "turn": _r(dec.turn_rate), "thrust": _r(dec.thrust, 2),
                "shoot": bool(dec.shoot),
            }
            csp = dec.csp_decision
            if csp is not None:
                d["csp"] = {
                    "ok": bool(csp.should_shoot),
                    "audit": {k: bool(v) for k, v in (csp.constraint_audit or {}).items()},
                    "why": csp.explanation,
                    "target": csp.target_track_id,
                    "aim": list(csp.intercept_point) if csp.intercept_point is not None else None,
                }
            mm = dec.minimax_decision
            if mm is not None:
                d["mm"] = {
                    "tactic": mm.chosen_tactic.name,
                    "u": _r(mm.utility_score),
                    "rival": mm.best_rival_response.name,
                    "nodes": int(mm.evaluated_nodes),
                    "pruned": int(mm.alpha_beta_pruned_nodes),
                    "target": mm.target_description,
                    "wp": [_r(mm.target_waypoint[0]), _r(mm.target_waypoint[1])],
                    "why": mm.explanation,
                    "searched": self.rival_bot is not None and 2 in eng.ships and eng.ships[2].is_alive,
                }
            plan = dec.astar_plan
            if plan is not None and self.plan_origin is not None:
                grid = plan.grid_cost_map
                d["astar"] = {
                    "found": bool(plan.is_path_found),
                    "path": [[_r(p[0]), _r(p[1])] for p in (plan.path_world or [])],
                    "goal": [_r(plan.subgoal_world[0]), _r(plan.subgoal_world[1])] if plan.subgoal_world else None,
                    "origin": [_r(self.plan_origin[0]), _r(self.plan_origin[1])],
                    "grid": ["".join("1" if v else "0" for v in row) for row in grid.tolist()] if grid is not None else None,
                }
            f = dec.facts
            if f is not None:
                tti = float(f.time_to_impact)
                d["facts"] = {
                    "tti": _r(tti, 2) if tti < 1e6 else None,
                    "closure": _r(f.closure_rate),
                    "nearest": _r(f.nearest_asteroid_dist) if f.nearest_asteroid_dist < 1e6 else None,
                    "imminent": bool(f.imminent_collision),
                    "hazard": [_r(f.imminent_hazard.position[0]), _r(f.imminent_hazard.position[1])] if f.imminent_hazard is not None else None,
                    "wall": bool(f.wall_proximity),
                }
            state["decision"] = d
        return state


_demo = None


def new_episode(seed, mode, params):
    global _demo
    _demo = LiveDemo(seed=seed, mode=mode, params=params)
    return json.dumps(_demo.snapshot())


def step(n=1):
    return _demo.step(n)
