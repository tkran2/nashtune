"""The backend-agnostic GRPO trainer (and the supervised baseline).

One model plays both seats. Each step:

1. Every ``refresh_every`` steps, extract the complete current policy and
   recompute exploitability, the opponent best responses and the exact action
   values Q(I, a) (see ``nashtune.rewards``). This is the expensive part for a
   language model, so it can be done less often than every step.
2. Build an update batch for the chosen reward mode:

   * ``br``: pick infosets (all, or a random subset), sample G actions per
     infoset from the old policy, reward each with Q(I, a), and normalize
     within the group.
   * ``exact``: the same, with the expectation over actions taken exactly
     instead of sampled (the infinite-group limit; lower variance).
   * ``selfplay``: deal hands, play G hands per deal with the policy in both
     seats, and reward each decision with the acting player's final payoff,
     normalized within the deal's group.
   * ``sft``: cross-entropy to an equilibrium strategy (the supervised baseline).
3. Take ``updates_per_batch`` optimizer steps on the clipped surrogate plus the
   exact KL penalty to the step-0 policy.

Checkpoints hold everything needed to resume bit-for-bit on the same machine.
"""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np

from .games import game_value, make_game, read_strategy, solve_to, write_strategy
from .grpo import BatchBuilder, entropy, expected_advantages, group_advantages, kl_categorical
from .policy.base import Policy
from .rewards import RefreshInfo, refresh, sample_deals, simulate

REWARD_MODES = ("br", "exact", "selfplay", "sft")
WEIGHTINGS = ("uniform", "reach", "cfreach")

LOG_COLUMNS = [
    "step", "exploitability_ma", "seat0_ma", "seat1_ma", "value", "mean_reward", "kl_ref", "legal_mass",
    "entropy", "avg_exploitability_ma", "batch_infosets", "samples", "skipped_groups", "loss", "clip_frac", "step_seconds",
    "refresh_seconds", "elapsed_seconds",
]


@dataclass
class TrainConfig:
    game: str = "kuhn"
    reward: str = "br"
    steps: int = 200
    group_size: int = 8
    infosets_per_step: int = 0  # 0 means every infoset, every step
    deals_per_step: int = 32  # selfplay only
    refresh_every: int = 1
    updates_per_batch: int = 1
    clip_eps: float = 0.2
    kl_coef: float = 0.0
    weighting: str = "uniform"
    adv_norm: str = "std"
    seed: int = 0
    checkpoint_every: int = 0
    sft_target: str = "cfr+"  # "cfr+" or a strategy file
    sft_target_ma: float = 0.1  # CFR+ is run until it is this close to equilibrium
    print_every: int = 10
    # Also track the reach-weighted average of the policies seen at refresh
    # steps (what CFR and fictitious play converge in) and log its exploitability.
    track_average: bool = False

    def validate(self) -> None:
        if self.reward not in REWARD_MODES:
            raise ValueError(f"reward must be one of {REWARD_MODES}")
        if self.weighting not in WEIGHTINGS:
            raise ValueError(f"weighting must be one of {WEIGHTINGS}")
        if self.adv_norm not in ("std", "none"):
            raise ValueError("adv_norm must be std or none")
        if self.group_size < 2 and self.reward in ("br", "selfplay"):
            raise ValueError("group_size must be at least 2 for sampled rewards")
        if self.steps < 0 or self.refresh_every < 1 or self.updates_per_batch < 1:
            raise ValueError("steps >= 0, refresh_every >= 1 and updates_per_batch >= 1 are required")

    @classmethod
    def from_dict(cls, data: dict) -> "TrainConfig":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names})


def _fmt(x) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    if isinstance(x, float):
        return f"{x:.10g}"
    return str(x)


class Trainer:
    def __init__(self, policy: Policy, config: TrainConfig, out_dir: str | os.PathLike | None = None,
                 resume: bool = True, echo=print) -> None:
        config.validate()
        self.policy = policy
        self.config = config
        self.game = policy.game
        if self.game.name != config.game:
            raise ValueError(f"policy is for {self.game.name} but config says {config.game}")
        self.out = Path(out_dir) if out_dir else None
        self.echo = echo
        self.rng = np.random.default_rng(config.seed)
        self.step = 0
        self.rows: list[dict] = []
        self.ref_logp: np.ndarray | None = None
        self.info: RefreshInfo | None = None
        self.legal_mass: np.ndarray | None = None
        self.elapsed = 0.0
        self._target: np.ndarray | None = None
        self._avg_num: np.ndarray | None = None
        self._avg_den: np.ndarray | None = None
        self._counts = np.array([i.action_count for i in self.game.infosets])
        if self.out:
            self.out.mkdir(parents=True, exist_ok=True)
            config_path = self.out / "config.json"
            if resume and self._latest() is not None:
                self.load_checkpoint(self._latest())
            elif not config_path.exists() or not resume:
                config_path.write_text(json.dumps({"train": asdict(config), "policy": policy.describe()}, indent=1) + "\n")

    # ------------------------------------------------------------- helpers
    def _slots(self, i: int) -> slice:
        info = self.game.infosets[int(i)]
        return slice(info.offset, info.offset + info.action_count)

    def sft_target(self) -> np.ndarray:
        if self._target is None:
            c = self.config
            if c.sft_target == "cfr+":
                self._target, _, _ = solve_to(self.game, c.sft_target_ma)
            else:
                self._target = read_strategy(self.game, c.sft_target)
        return self._target

    def _do_refresh(self) -> dict:
        start = time.time()
        ext = self.policy.extract()
        strategy = ext.strategy
        logs = np.log(np.maximum(strategy, 1e-300))
        if self.ref_logp is None:
            self.ref_logp = logs.copy()
        self.info = refresh(self.game, strategy, need_values=self.config.reward in ("br", "exact"))
        self.legal_mass = ext.legal_mass
        seat0, seat1 = self.info.seat_milli(self.game.name)
        extra = {}
        if self.config.track_average:
            extra["avg_exploitability_ma"] = self._update_average(strategy)
        kls, ents = [], []
        for info in self.game.infosets:
            s = self._slots(info.index)
            kls.append(kl_categorical(logs[s], self.ref_logp[s]))
            ents.append(entropy(logs[s]))
        return {
            "exploitability_ma": self.info.expl.milli, "seat0_ma": seat0, "seat1_ma": seat1,
            "value": self.info.expl.value, "kl_ref": float(np.mean(kls)), "entropy": float(np.mean(ents)),
            "legal_mass": float(np.mean(ext.legal_mass)) if ext.legal_mass is not None else 1.0,
            "refresh_seconds": time.time() - start, **extra,
        }

    def _update_average(self, strategy: np.ndarray) -> float:
        from .games.analysis import exploitability, own_reach

        g = self.game
        weight = np.zeros(g.action_slots)
        for player in (0, 1):
            mask = g.slot_player == player
            weight[mask] = own_reach(g, strategy, player)[g.slot_infoset][mask]
        if self._avg_num is None:
            self._avg_num = np.zeros(g.action_slots)
            self._avg_den = np.zeros(g.action_slots)
        self._avg_num += weight * strategy
        self._avg_den += weight
        avg = np.where(self._avg_den > 0, self._avg_num / np.where(self._avg_den > 0, self._avg_den, 1.0), strategy)
        avg = g.normalize(avg)
        return exploitability(g, avg).milli

    def _select(self) -> np.ndarray:
        n = self.game.num_infosets
        k = self.config.infosets_per_step
        if k <= 0 or k >= n:
            return np.arange(n)
        return np.sort(self.rng.choice(n, size=k, replace=False))

    def _weights(self, ids: np.ndarray) -> np.ndarray:
        w_mode = self.config.weighting
        if w_mode == "uniform" or self.info is None:
            w = np.ones(len(ids))
        elif w_mode == "reach":
            w = self.info.reach_weight[ids].copy()
        else:
            w = self.info.cf_reach[ids].copy()
        if w.sum() <= 0:
            w = np.ones(len(ids))
        return w / w.sum()

    def _old_logs(self, ids: np.ndarray, refreshed: bool) -> list[np.ndarray]:
        if refreshed and self.info is not None:
            # The policy has not changed since the extraction: reuse it.
            out = []
            for i in ids:
                p = self.info.strategy[self._slots(i)]
                out.append(np.log(np.maximum(p, 1e-300)))
            return out
        return self.policy.logpi(ids)

    # ------------------------------------------------------------ batches
    def _batch_values(self, refreshed: bool):
        c = self.config
        ids = self._select()
        ids = ids[self.info.defined[ids]]  # the best-responding opponent never lets the others happen
        weights = self._weights(ids)
        logs = self._old_logs(ids, refreshed)
        builder = BatchBuilder(self._counts, c.clip_eps)
        skipped, samples, reward_sum = 0, 0, 0.0
        for i, w, lp in zip(ids, weights, logs):
            q = self.info.q[self._slots(i)]
            pi = np.exp(lp)
            pi = pi / pi.sum()
            builder.set_old(int(i), lp)
            builder.set_ref(int(i), self.ref_logp[self._slots(i)])
            if c.kl_coef:
                builder.add_kl(int(i), c.kl_coef * w)
            if c.reward == "br":
                actions = self.rng.choice(len(pi), size=c.group_size, p=pi)
                rewards = q[actions]
                reward_sum += w * float(rewards.mean())
                samples += c.group_size
                adv, used = group_advantages(rewards, c.adv_norm)
                if not used:
                    skipped += 1
                    continue
                for a, A in zip(actions, adv):
                    builder.add_sample(int(i), int(a), float(A), w / c.group_size)
            else:  # exact
                reward_sum += w * float(pi @ q)
                adv, used = expected_advantages(q, pi, c.adv_norm)
                if not used:
                    skipped += 1
                    continue
                for a in range(len(pi)):
                    builder.add_sample(int(i), a, float(adv[a]), w * float(pi[a]))
        batch = builder.build()
        return batch, {"batch_infosets": len(ids), "samples": samples, "skipped_groups": skipped,
                       "mean_reward": reward_sum}

    def _batch_selfplay(self, refreshed: bool):
        c = self.config
        deals = sample_deals(self.game, c.deals_per_step, self.rng)

        def probs_fn(ids):
            return [np.exp(lp) for lp in self._old_logs(np.array(ids, dtype=np.int64), refreshed)]

        roll = simulate(self.game, deals, c.group_size, probs_fn, self.rng)
        builder = BatchBuilder(self._counts, c.clip_eps)
        for i, p in roll.probs.items():
            builder.set_old(i, np.log(np.maximum(p, 1e-300)))
            builder.set_ref(i, self.ref_logp[self._slots(i)])
        if c.kl_coef:
            for i in roll.probs:
                builder.add_kl(i, c.kl_coef / len(roll.probs))
        adv = np.zeros((2, len(roll.payoff)))
        skipped = 0
        for g in range(c.deals_per_step):
            members = np.flatnonzero(roll.deal == g)
            a0, used = group_advantages(roll.payoff[members], c.adv_norm)
            if not used:
                skipped += 1
                continue
            adv[0, members] = a0
            adv[1, members] = -a0
        weight = 1.0 / (c.deals_per_step * c.group_size)
        rewards = []
        for r, player, i, a in roll.decisions:
            rewards.append(roll.payoff[r] if player == 0 else -roll.payoff[r])
            if adv[player, r] != 0.0:
                builder.add_sample(i, a, float(adv[player, r]), weight)
        batch = builder.build()
        return batch, {"batch_infosets": len(roll.probs), "samples": len(roll.decisions),
                       "skipped_groups": skipped, "mean_reward": float(np.mean(rewards)) if rewards else 0.0}

    def _batch_sft(self):
        target = self.sft_target()
        ids = self._select()
        builder = BatchBuilder(self._counts, self.config.clip_eps)
        w = 1.0 / len(ids)
        for i in ids:
            s = self._slots(i)
            builder.set_old(int(i), np.zeros(s.stop - s.start))  # unused: no policy-gradient term
            builder.set_ref(int(i), self.ref_logp[s])
            builder.add_ce(int(i), target[s], w)
        return builder.build(), {"batch_infosets": len(ids), "samples": 0, "skipped_groups": 0, "mean_reward": float("nan")}

    # --------------------------------------------------------------- steps
    def train_step(self) -> dict:
        c = self.config
        start = time.time()
        row: dict = {"step": self.step}
        refreshed = self.info is None or self.step % c.refresh_every == 0
        if refreshed:
            row.update(self._do_refresh())
        if c.reward in ("br", "exact"):
            batch, bstats = self._batch_values(refreshed)
        elif c.reward == "selfplay":
            batch, bstats = self._batch_selfplay(refreshed)
        else:
            batch, bstats = self._batch_sft()
        row.update(bstats)
        # Called even when every group was skipped: an optimizer with momentum
        # still moves, as it would in any GRPO implementation (zero loss, zero
        # gradient, optimizer step).
        row.update(self.policy.update(batch, c.updates_per_batch))
        self.step += 1
        row["step_seconds"] = time.time() - start
        self.elapsed += row["step_seconds"]
        row["elapsed_seconds"] = self.elapsed
        self.rows.append(row)
        return row

    def evaluate_final(self) -> dict:
        start = time.time()
        row = {"step": self.step}
        row.update(self._do_refresh())
        self.elapsed += time.time() - start
        row["step_seconds"] = time.time() - start
        row["elapsed_seconds"] = self.elapsed
        self.rows.append(row)
        return row

    def run(self) -> dict:
        c = self.config
        if self.rows and self.rows[-1].get("final"):
            self.echo(f"[nashtune] {self.out} is already complete ({self.step} steps)")
            return self.rows[-1]
        first = self.step
        while self.step < c.steps:
            row = self.train_step()
            done = self.step - first
            if done == 3 or (c.print_every and (self.step % c.print_every == 0 or self.step == c.steps)):
                rate = self.elapsed / max(self.step, 1)
                eta = rate * (c.steps - self.step)
                msg = f"step {self.step:5d}/{c.steps}  {rate:6.2f} s/step  ETA {eta / 60:6.1f} min"
                if not math.isnan(row.get("exploitability_ma", float("nan"))):
                    msg += f"  exploitability {row['exploitability_ma']:9.2f} mA"
                last = self._last_measured()
                if last is not None and "exploitability_ma" not in row:
                    msg += f"  (last measured {last:9.2f} mA)"
                self.echo(msg)
            if c.checkpoint_every and self.step % c.checkpoint_every == 0 and self.out:
                self.save_checkpoint()
        final = self.evaluate_final()
        final["final"] = 1
        self.echo(f"final exploitability {final['exploitability_ma']:.2f} mA after {self.step} steps "
                  f"({self.elapsed / 60:.1f} min)")
        if self.out:
            self.write_outputs()
            if c.checkpoint_every:
                self.save_checkpoint()
        return final

    def _last_measured(self):
        for row in reversed(self.rows):
            if "exploitability_ma" in row:
                return row["exploitability_ma"]
        return None

    # -------------------------------------------------------------- output
    def write_log(self) -> None:
        if not self.out:
            return
        path = self.out / "log.csv"
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(LOG_COLUMNS)
            for row in self.rows:
                w.writerow([_fmt(row.get(k)) for k in LOG_COLUMNS])
        os.replace(tmp, path)

    def write_outputs(self) -> None:
        self.write_log()
        assert self.info is not None
        write_strategy(self.game, self.info.strategy, self.out / "final_strategy.txt")
        first = next((r for r in self.rows if "exploitability_ma" in r), None)
        last = self.rows[-1]
        summary = {
            "game": self.game.name, "reward": self.config.reward, "steps": self.step,
            "initial_exploitability_ma": first["exploitability_ma"] if first else None,
            "final_exploitability_ma": last["exploitability_ma"],
            "final_seat0_ma": last["seat0_ma"], "final_seat1_ma": last["seat1_ma"],
            "final_value": last["value"], "final_kl_ref": last["kl_ref"], "final_legal_mass": last["legal_mass"],
            "game_value": game_value(self.game.name), "elapsed_seconds": self.elapsed,
            "config": asdict(self.config), "policy": self.policy.describe(),
        }
        (self.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
        self.policy.save(self.out / "policy")

    # --------------------------------------------------------- checkpoints
    def _latest(self) -> Path | None:
        if not self.out:
            return None
        pointer = self.out / "checkpoints" / "latest"
        if pointer.exists():
            target = self.out / "checkpoints" / pointer.read_text().strip()
            if (target / "trainer.json").exists():
                return target
        return None

    def save_checkpoint(self) -> Path:
        assert self.out is not None
        root = self.out / "checkpoints"
        root.mkdir(exist_ok=True)
        name = f"step_{self.step:06d}"
        tmp = root / (name + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir()
        meta = {"step": self.step, "elapsed": self.elapsed, "rows": self.rows,
                "rng": self.rng.bit_generator.state, "config": asdict(self.config)}
        (tmp / "trainer.json").write_text(json.dumps(meta, default=_json_default) + "\n")
        arrays = {"ref_logp": self.ref_logp if self.ref_logp is not None else np.zeros(0)}
        if self.info is not None:
            arrays.update({"info_strategy": self.info.strategy, "info_q": self.info.q, "info_defined": self.info.defined,
                           "info_cf_reach": self.info.cf_reach, "info_own_reach": self.info.own_reach,
                           "info_br0": self.info.br_profile[0], "info_br1": self.info.br_profile[1],
                           "info_expl": np.array([self.info.expl.value, self.info.expl.best_vs_p1, self.info.expl.best_vs_p0])})
        if self.legal_mass is not None:
            arrays["legal_mass"] = self.legal_mass
        if self._avg_num is not None:
            arrays["avg_num"], arrays["avg_den"] = self._avg_num, self._avg_den
        np.savez(tmp / "trainer_arrays.npz", **arrays)
        save_policy_state(self.policy, tmp)
        final = root / name
        if final.exists():
            shutil.rmtree(final)
        os.replace(tmp, final)
        (root / "latest.tmp").write_text(name + "\n")
        os.replace(root / "latest.tmp", root / "latest")
        # Keep the two most recent checkpoints.
        for old in sorted(p for p in root.glob("step_*") if p.is_dir() and p.name != name)[:-1]:
            shutil.rmtree(old, ignore_errors=True)
        self.write_log()
        return final

    def load_checkpoint(self, path: Path) -> None:
        from .games.analysis import Exploitability

        meta = json.loads((path / "trainer.json").read_text())
        self.step = int(meta["step"])
        self.elapsed = float(meta["elapsed"])
        self.rows = meta["rows"]
        self.rng.bit_generator.state = meta["rng"]
        arrays = np.load(path / "trainer_arrays.npz")
        self.ref_logp = arrays["ref_logp"] if arrays["ref_logp"].size else None
        if "info_strategy" in arrays:
            e = arrays["info_expl"]
            self.info = RefreshInfo(
                strategy=arrays["info_strategy"], expl=Exploitability(float(e[0]), float(e[1]), float(e[2])),
                q=arrays["info_q"], defined=arrays["info_defined"].astype(bool), cf_reach=arrays["info_cf_reach"],
                own_reach=arrays["info_own_reach"], br_profile=[arrays["info_br0"], arrays["info_br1"]])
        self.legal_mass = arrays["legal_mass"] if "legal_mass" in arrays else None
        if "avg_num" in arrays:
            self._avg_num, self._avg_den = arrays["avg_num"], arrays["avg_den"]
        load_policy_state(self.policy, path)
        self.echo(f"[nashtune] resumed from {path} at step {self.step}")


def _json_default(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(type(x))


def save_policy_state(policy: Policy, directory: Path) -> None:
    saver = getattr(policy, "save_state", None)
    if saver is not None:
        saver(directory)
        return
    state = policy.state_dict()
    flat = {}
    meta = {}
    for k, v in state.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                if isinstance(v2, np.ndarray):
                    flat[f"{k}.{k2}"] = v2
                else:
                    meta[f"{k}.{k2}"] = v2
        elif isinstance(v, np.ndarray):
            flat[k] = v
        else:
            meta[k] = v
    np.savez(directory / "policy_arrays.npz", **flat)
    (directory / "policy_meta.json").write_text(json.dumps(meta, default=_json_default) + "\n")


def load_policy_state(policy: Policy, directory: Path) -> None:
    loader = getattr(policy, "load_state", None)
    if loader is not None:
        loader(directory)
        return
    arrays = np.load(directory / "policy_arrays.npz")
    meta = json.loads((directory / "policy_meta.json").read_text())
    state: dict = {}
    for k in list(arrays.keys()) + list(meta.keys()):
        value = arrays[k] if k in arrays else meta[k]
        if "." in k:
            a, b = k.split(".", 1)
            state.setdefault(a, {})[b] = value
        else:
            state[k] = value
    policy.load_state_dict(state)


def make_tabular_trainer(config: TrainConfig, lr: float = 0.05, init: str = "random", init_scale: float = 1.0,
                         out_dir=None, resume: bool = True, echo=print) -> Trainer:
    from .policy.tabular import TabularPolicy

    game = make_game(config.game)
    policy = TabularPolicy(game, init=init, init_scale=init_scale, seed=config.seed, lr=lr)
    return Trainer(policy, config, out_dir=out_dir, resume=resume, echo=echo)


def quiet(*_args, **_kwargs) -> None:
    pass


def stderr_echo(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)
