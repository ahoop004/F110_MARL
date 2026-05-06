"""Per-episode console logging callback for SB3 MARL training.

Prints one compact line per episode and a rolling summary every N episodes.
Works by reading the info dict that SB3RoleWrapper injects into each step's
`infos` payload via the Monitor wrapper.

Expected info keys (set by SB3RoleWrapper):
    outcome        — str  e.g. "timeout", "collision", "lap_complete"
    focal_agent    — str  e.g. "car_0"
    role_episode   — int  monotonically increasing episode counter
    is_success     — bool
    reward_components — dict (optional, only if reward strategy is set)
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, Optional, Deque

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

try:
    from rich.console import Console
    from rich.text import Text
    _RICH = True
except ImportError:
    _RICH = False


# ---------------------------------------------------------------------------
# ANSI fallback when rich is unavailable
# ---------------------------------------------------------------------------

_ANSI = {
    "reset":  "\033[0m",
    "bold":   "\033[1m",
    "dim":    "\033[2m",
    "green":  "\033[32m",
    "yellow": "\033[33m",
    "red":    "\033[31m",
    "cyan":   "\033[36m",
    "blue":   "\033[34m",
    "white":  "\033[37m",
}


def _c(text: str, *styles: str) -> str:
    codes = "".join(_ANSI.get(s, "") for s in styles)
    return f"{codes}{text}{_ANSI['reset']}" if codes else text


# ---------------------------------------------------------------------------
# Outcome metadata
# ---------------------------------------------------------------------------

_OUTCOME_META: Dict[str, Dict[str, str]] = {
    "timeout":      {"short": "TIMEOUT ", "style": "yellow"},
    "collision":    {"short": "COLLIDE ", "style": "red"},
    "lap_complete": {"short": "LAP  OK ", "style": "green"},
    "complete":     {"short": "COMPLETE", "style": "green"},
    "self_crash":   {"short": "CRASH   ", "style": "red"},
    "target_crash": {"short": "T-CRASH ", "style": "green"},
    "idle_stop":    {"short": "IDLE    ", "style": "dim yellow"},
    "truncated":    {"short": "TRUNC   ", "style": "dim"},
    "unknown":      {"short": "UNKNOWN ", "style": "dim"},
}

def _outcome_meta(outcome: str) -> Dict[str, str]:
    key = str(outcome).lower().strip()
    return _OUTCOME_META.get(key, {"short": key[:8].upper().ljust(8), "style": "white"})


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------

class SB3ConsoleCallback(BaseCallback):
    """Print per-episode stats to the terminal during SB3 MARL training.

    Args:
        log_every:
            Print a full episode line every N episodes (default 1 = every ep).
        summary_every:
            Print a rolling summary row every N episodes (default 25).
        window:
            Rolling-window size for reward / success / steps statistics.
        quiet:
            Suppress all output (no-op mode).
        show_reward_components:
            Include reward-component breakdown in the summary rows.
    """

    def __init__(
        self,
        log_every: int = 1,
        summary_every: int = 25,
        window: int = 100,
        quiet: bool = False,
        show_reward_components: bool = False,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.log_every = max(1, int(log_every))
        self.summary_every = max(1, int(summary_every))
        self.window = max(1, int(window))
        self.quiet = quiet
        self.show_reward_components = show_reward_components

        self._ep: int = 0
        self._rewards: Deque[float] = deque(maxlen=self.window)
        self._successes: Deque[bool] = deque(maxlen=self.window)
        self._steps: Deque[int] = deque(maxlen=self.window)
        self._outcomes: Deque[str] = deque(maxlen=self.window)
        self._reward_components: Deque[Dict[str, float]] = deque(maxlen=self.window)

        self._current_reward: float = 0.0
        self._current_steps: int = 0
        self._last_map: Optional[str] = None

        if _RICH:
            self._console = Console(highlight=False)
        else:
            self._console = None

    # ------------------------------------------------------------------
    # SB3 callback interface
    # ------------------------------------------------------------------

    def _on_step(self) -> bool:
        if self.quiet:
            return True

        # Accumulate reward
        rewards = self.locals.get("rewards")
        if rewards is not None:
            r = float(rewards[0]) if hasattr(rewards, "__len__") else float(rewards)
            self._current_reward += r
            self._current_steps += 1

        # Check for episode end
        done = self._is_done()
        if not done:
            return True

        info = self._get_info()
        self._process_episode(info)
        self._current_reward = 0.0
        self._current_steps = 0
        return True

    # ------------------------------------------------------------------
    # Episode processing
    # ------------------------------------------------------------------

    def _process_episode(self, info: Dict[str, Any]) -> None:
        self._ep += 1

        outcome = str(info.get("outcome", "unknown")).lower()
        focal = str(info.get("focal_agent", "?"))
        success = bool(info.get("is_success", False))
        components = info.get("reward_components") or {}

        # Try to get map name from the wrapped env
        map_name = self._get_map_name()

        self._rewards.append(self._current_reward)
        self._successes.append(success)
        self._steps.append(self._current_steps)
        self._outcomes.append(outcome)
        if components:
            self._reward_components.append(dict(components))

        # Per-episode line
        if self._ep % self.log_every == 0:
            self._print_episode_line(
                ep=self._ep,
                focal=focal,
                outcome=outcome,
                reward=self._current_reward,
                steps=self._current_steps,
                map_name=map_name,
                success=success,
            )

        # Rolling summary
        if self._ep % self.summary_every == 0:
            self._print_summary()

    # ------------------------------------------------------------------
    # Printing
    # ------------------------------------------------------------------

    def _print_episode_line(
        self,
        ep: int,
        focal: str,
        outcome: str,
        reward: float,
        steps: int,
        map_name: Optional[str],
        success: bool,
    ) -> None:
        meta = _outcome_meta(outcome)
        short = meta["short"]
        style = meta["style"]

        map_str = (map_name or "?")[:14].ljust(14)
        focal_str = focal.ljust(6)
        r_sign = "+" if reward >= 0 else ""

        if _RICH:
            t = Text()
            t.append(f"ep={ep:5d}  ", style="dim")
            t.append(f"{focal_str}  ", style="bold cyan")
            t.append(f"{map_str}  ", style="dim")
            t.append(f"{short}  ", style=style)
            t.append(f"r={r_sign}{reward:7.2f}  ", style="bold" if success else "")
            t.append(f"steps={steps:6d}", style="dim")
            self._console.print(t)
        else:
            style_fn = {
                "green": lambda s: _c(s, "green", "bold"),
                "red": lambda s: _c(s, "red"),
                "yellow": lambda s: _c(s, "yellow"),
                "dim": lambda s: _c(s, "dim"),
                "dim yellow": lambda s: _c(s, "dim", "yellow"),
                "white": lambda s: s,
            }.get(style, lambda s: s)

            line = (
                f"{_c(f'ep={ep:5d}', 'dim')}  "
                f"{_c(focal_str, 'cyan')}  "
                f"{_c(map_str, 'dim')}  "
                f"{style_fn(short)}  "
                f"r={r_sign}{reward:7.2f}  "
                f"{_c(f'steps={steps:6d}', 'dim')}"
            )
            print(line)

    def _print_summary(self) -> None:
        if not self._rewards:
            return

        n = len(self._rewards)
        avg_r = float(np.mean(self._rewards))
        avg_s = float(np.mean(self._steps))
        suc_rate = float(np.mean(self._successes))

        # Outcome breakdown
        outcome_counts: Dict[str, int] = {}
        for o in self._outcomes:
            outcome_counts[o] = outcome_counts.get(o, 0) + 1
        top_outcomes = sorted(outcome_counts.items(), key=lambda x: -x[1])[:4]
        outcome_str = "  ".join(
            f"{_outcome_meta(o)['short'].strip()}={c}" for o, c in top_outcomes
        )

        # Optional reward component means
        comp_str = ""
        if self.show_reward_components and self._reward_components:
            all_keys: set = set()
            for d in self._reward_components:
                all_keys.update(d.keys())
            comp_parts = []
            for k in sorted(all_keys):
                vals = [d[k] for d in self._reward_components if k in d]
                if vals:
                    comp_parts.append(f"{k}={np.mean(vals):+.3f}")
            comp_str = "  " + "  ".join(comp_parts) if comp_parts else ""

        line = (
            f"── ep={self._ep:5d}  last={n}ep  "
            f"avg_r={avg_r:+.2f}  avg_steps={avg_s:.0f}  "
            f"success={suc_rate:.1%}  [{outcome_str}]{comp_str}"
        )

        if _RICH:
            self._console.rule(line, style="dim blue")
        else:
            print(_c(f"\n{line}\n", "blue", "bold"))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_done(self) -> bool:
        dones = self.locals.get("dones")
        if dones is not None:
            return bool(dones[0]) if hasattr(dones, "__len__") else bool(dones)
        term = self.locals.get("terminateds")
        trunc = self.locals.get("truncateds")
        if term is not None or trunc is not None:
            t = bool(term[0]) if hasattr(term, "__len__") else bool(term or False)
            u = bool(trunc[0]) if hasattr(trunc, "__len__") else bool(trunc or False)
            return t or u
        return False

    def _get_info(self) -> Dict[str, Any]:
        infos = self.locals.get("infos")
        if isinstance(infos, list) and infos:
            return infos[0] or {}
        if isinstance(infos, dict):
            return infos
        return {}

    def _get_map_name(self) -> Optional[str]:
        """Pull current map name from the wrapped env (best-effort)."""
        try:
            env = self.training_env
            # Unwrap Monitor → SB3RoleWrapper → F110ParallelEnv
            inner = getattr(env, "envs", [env])[0]
            for _ in range(5):
                if hasattr(inner, "map_name"):
                    raw = getattr(inner, "map_name", None)
                    if raw and raw != self._last_map:
                        self._last_map = raw
                    return self._last_map
                inner = getattr(inner, "env", inner)
                if inner is env:
                    break
        except Exception:
            pass
        return self._last_map
