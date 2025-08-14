# # logger.py
# import csv
# import json
# import math
# import time
# import shutil
# from dataclasses import dataclass, asdict
# from pathlib import Path
# from typing import Any, Dict, Optional, List


# def _now_stamp() -> str:
#     return time.strftime("%Y-%m-%d_%H-%M-%S")


# @dataclass
# class EpisodeLog:
#     episode: int
#     reward: float
#     loss: Optional[float] = None
#     epsilon: Optional[float] = None
#     steps: Optional[int] = None
#     duration_sec: Optional[float] = None
#     info: Optional[Dict[str, Any]] = None  # free-form extras


# class TrainingLogger:
#     """
#     Saves:
#       - metrics.csv: per-episode logs (episode, reward, loss, epsilon, steps, duration)
#       - metrics.jsonl: same data in JSONL for easy programmatic reads
#       - config.json: your hyperparameters / run config
#       - best.ckpt: best model checkpoint by moving-average reward
#       - last.ckpt: last (latest) model checkpoint
#       - plot.png (optional): quick progress plot if matplotlib is available
#     """

#     def __init__(
#         self,
#         run_dir: Optional[str] = None,
#         project: str = "RL_PacMan",
#         tag: Optional[str] = None,
#         avg_window: int = 20,
#         keep_n_last: int = 3,  # keep a few rolling "last_*.ckpt" copies
#     ):
#         stamp = _now_stamp()
#         tag = f"-{tag}" if tag else ""
#         self.dir = Path(run_dir) if run_dir else Path(f"runs/{project}/{stamp}{tag}")
#         self.dir.mkdir(parents=True, exist_ok=True)

#         self.csv_path = self.dir / "metrics.csv"
#         self.jsonl_path = self.dir / "metrics.jsonl"
#         self.config_path = self.dir / "config.json"
#         self.best_ckpt = self.dir / "best.ckpt"
#         self.last_ckpt = self.dir / "last.ckpt"
#         self.keep_n_last = keep_n_last

#         self.avg_window = max(1, int(avg_window))
#         self._recent_rewards: List[float] = []
#         self._best_ma: float = -math.inf
#         self._step_counter = 0

#         # Initialize CSV header
#         if not self.csv_path.exists():
#             with self.csv_path.open("w", newline="") as f:
#                 writer = csv.writer(f)
#                 writer.writerow(
#                     [
#                         "episode",
#                         "reward",
#                         "loss",
#                         "epsilon",
#                         "steps",
#                         "duration_sec",
#                         f"ma_reward_{self.avg_window}",
#                     ]
#                 )

#         # JSONL file is append-only; create if missing
#         if not self.jsonl_path.exists():
#             self.jsonl_path.touch()

#     def save_config(self, cfg: Dict[str, Any]) -> None:
#         """Save your hyperparameters once at the beginning."""
#         with self.config_path.open("w") as f:
#             json.dump(cfg, f, indent=2, sort_keys=True)

#     def _moving_avg(self) -> float:
#         if not self._recent_rewards:
#             return float("nan")
#         return sum(self._recent_rewards) / len(self._recent_rewards)

#     def log_episode(self, entry: EpisodeLog) -> float:
#         """Append one episode’s metrics. Returns current moving-average reward."""
#         # Track reward window for moving-average
#         self._recent_rewards.append(entry.reward)
#         if len(self._recent_rewards) > self.avg_window:
#             self._recent_rewards.pop(0)
#         ma = self._moving_avg()

#         # CSV
#         with self.csv_path.open("a", newline="") as f:
#             writer = csv.writer(f)
#             writer.writerow(
#                 [
#                     entry.episode,
#                     entry.reward,
#                     entry.loss if entry.loss is not None else "",
#                     entry.epsilon if entry.epsilon is not None else "",
#                     entry.steps if entry.steps is not None else "",
#                     entry.duration_sec if entry.duration_sec is not None else "",
#                     ma,
#                 ]
#             )

#         # JSONL
#         row = asdict(entry)
#         row[f"ma_reward_{self.avg_window}"] = ma
#         with self.jsonl_path.open("a") as f:
#             f.write(json.dumps(row) + "\n")

#         return ma

#     def maybe_checkpoint(
#         self,
#         model: Any,
#         optimizer: Optional[Any] = None,
#         global_step: Optional[int] = None,
#         extra: Optional[Dict[str, Any]] = None,
#         prefer_torch: bool = True,
#         is_best_if_ma_improves: bool = True,
#     ) -> Dict[str, Any]:
#         """
#         Save 'last.ckpt' every call; update 'best.ckpt' if the moving-average improves.
#         Works with PyTorch (preferred) or Keras (fallback).
#         """
#         snapshot = {
#             "global_step": global_step,
#             "timestamp": _now_stamp(),
#             "ma_reward": self._moving_avg(),
#             "extra": extra or {},
#         }

#         # Try PyTorch first
#         saved_with_torch = False
#         if prefer_torch:
#             try:
#                 import torch

#                 state = {
#                     "model": (
#                         model.state_dict() if hasattr(model, "state_dict") else None
#                     ),
#                     "optimizer": (
#                         optimizer.state_dict()
#                         if (optimizer is not None and hasattr(optimizer, "state_dict"))
#                         else None
#                     ),
#                     "meta": snapshot,
#                 }
#                 torch.save(state, self.last_ckpt)
#                 saved_with_torch = True

#                 # keep rotating last_X.ckpt for a tiny history
#                 if self.keep_n_last > 0:
#                     idx = (self._step_counter % self.keep_n_last) + 1
#                     torch.save(state, self.dir / f"last_{idx}.ckpt")
#                     self._step_counter += 1

#             except Exception:
#                 saved_with_torch = False

#         # Fallback: generic/keras save
#         if not saved_with_torch:
#             # Save model in a general way
#             # If Keras: model.save(self.last_ckpt_dir)
#             last_dir = self.dir / "last_model"
#             if last_dir.exists():
#                 shutil.rmtree(last_dir)
#             last_dir.mkdir(parents=True, exist_ok=True)

#             # Try keras save()
#             try:
#                 model.save(last_dir.as_posix())  # tf/keras
#             except Exception:
#                 # Final fallback: pickle-ish torchless state (may not be restorable)
#                 with (last_dir / "model.json").open("w") as f:
#                     json.dump(
#                         {
#                             "warning": "Generic save; please provide a framework-specific saver."
#                         },
#                         f,
#                     )

#             with (last_dir / "meta.json").open("w") as f:
#                 json.dump(snapshot, f, indent=2, sort_keys=True)

#         # Update best by moving-average reward
#         if is_best_if_ma_improves:
#             ma = snapshot["ma_reward"]
#             if ma is not None and not math.isnan(ma) and ma > self._best_ma:
#                 self._best_ma = ma
#                 if saved_with_torch:
#                     # copy last.ckpt -> best.ckpt
#                     shutil.copy2(self.last_ckpt, self.best_ckpt)
#                 else:
#                     # copy directory
#                     best_dir = self.dir / "best_model"
#                     if best_dir.exists():
#                         shutil.rmtree(best_dir)
#                     shutil.copytree(self.dir / "last_model", best_dir)

#         return snapshot

#     def save_plot(self) -> None:
#         """Optional: quick PNG plot of reward and moving-average."""
#         try:
#             import pandas as pd
#             import matplotlib.pyplot as plt

#             df = pd.read_csv(self.csv_path)
#             fig = plt.figure()
#             ax = fig.add_subplot(111)
#             ax.plot(df["episode"], df["reward"], label="reward")
#             ma_col = [c for c in df.columns if c.startswith("ma_reward_")]
#             if ma_col:
#                 ax.plot(df["episode"], df[ma_col[0]], label=ma_col[0])
#             ax.set_xlabel("Episode")
#             ax.set_ylabel("Reward")
#             ax.legend()
#             fig.tight_layout()
#             fig.savefig(self.dir / "plot.png")
#             plt.close(fig)
#         except Exception:
#             # plotting is optional; ignore errors to avoid breaking training
#             pass

import csv
import json
import math
import time
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, List


def _now_stamp() -> str:
    return time.strftime("%Y-%m-%d_%H-%M-%S")


@dataclass
class EpisodeLog:
    episode: int
    reward: float
    loss: Optional[float] = None
    epsilon: Optional[float] = None
    steps: Optional[int] = None
    duration_sec: Optional[float] = None
    info: Optional[Dict[str, Any]] = None


class TrainingLogger:
    def __init__(
        self,
        run_dir: Optional[str] = None,
        project: str = "RL_PacMan",
        tag: Optional[str] = None,
        avg_window: int = 20,
        keep_n_last: int = 3,
    ):
        stamp = _now_stamp()
        tag = f"-{tag}" if tag else ""
        self.dir = Path(run_dir) if run_dir else Path(f"runs/{project}/{stamp}{tag}")
        self.dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.dir / "metrics.csv"
        self.jsonl_path = self.dir / "metrics.jsonl"
        self.config_path = self.dir / "config.json"
        self.best_ckpt = self.dir / "best.ckpt"
        self.last_ckpt = self.dir / "last.ckpt"
        self.keep_n_last = keep_n_last

        self.avg_window = max(1, int(avg_window))
        self._recent_rewards: List[float] = []
        self._best_ma: float = -math.inf
        self._step_counter = 0

        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="") as f:
                csv.writer(f).writerow(
                    [
                        "episode",
                        "reward",
                        "loss",
                        "epsilon",
                        "steps",
                        "duration_sec",
                        f"ma_reward_{self.avg_window}",
                    ]
                )
        if not self.jsonl_path.exists():
            self.jsonl_path.touch()

    def save_config(self, cfg: Dict[str, Any]) -> None:
        with self.config_path.open("w") as f:
            json.dump(cfg, f, indent=2, sort_keys=True)

    def _moving_avg(self) -> float:
        return (
            sum(self._recent_rewards) / len(self._recent_rewards)
            if self._recent_rewards
            else float("nan")
        )

    def log_episode(self, entry: EpisodeLog) -> float:
        self._recent_rewards.append(entry.reward)
        if len(self._recent_rewards) > self.avg_window:
            self._recent_rewards.pop(0)
        ma = self._moving_avg()

        with self.csv_path.open("a", newline="") as f:
            csv.writer(f).writerow(
                [
                    entry.episode,
                    entry.reward,
                    entry.loss or "",
                    entry.epsilon or "",
                    entry.steps or "",
                    entry.duration_sec or "",
                    ma,
                ]
            )
        row = asdict(entry)
        row[f"ma_reward_{self.avg_window}"] = ma
        with self.jsonl_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        return ma

    def maybe_checkpoint(
        self,
        model: Any,
        optimizer: Optional[Any] = None,
        global_step: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
        prefer_torch: bool = True,
        is_best_if_ma_improves: bool = True,
    ) -> Dict[str, Any]:
        snapshot = {
            "global_step": global_step,
            "timestamp": _now_stamp(),
            "ma_reward": self._moving_avg(),
            "extra": extra or {},
        }
        saved_with_torch = False
        if prefer_torch:
            try:
                import torch

                state = {
                    "model": (
                        model.state_dict() if hasattr(model, "state_dict") else None
                    ),
                    "optimizer": (
                        optimizer.state_dict()
                        if (optimizer is not None and hasattr(optimizer, "state_dict"))
                        else None
                    ),
                    "meta": snapshot,
                }
                torch.save(state, self.last_ckpt)
                saved_with_torch = True
                if self.keep_n_last > 0:
                    idx = (self._step_counter % self.keep_n_last) + 1
                    torch.save(state, self.dir / f"last_{idx}.ckpt")
                    self._step_counter += 1
            except Exception:
                saved_with_torch = False
        if not saved_with_torch:
            last_dir = self.dir / "last_model"
            if last_dir.exists():
                shutil.rmtree(last_dir)
            last_dir.mkdir(parents=True, exist_ok=True)
            try:
                model.save(last_dir.as_posix())
            except Exception:
                with (last_dir / "model.json").open("w") as f:
                    json.dump(
                        {"warning": "Generic save; provide framework-specific saver."},
                        f,
                    )
            with (last_dir / "meta.json").open("w") as f:
                json.dump(snapshot, f, indent=2, sort_keys=True)
        if is_best_if_ma_improves:
            ma = snapshot["ma_reward"]
            if (
                ma is not None
                and not math.isnan(ma)
                and ma > getattr(self, "_best_ma", -math.inf)
            ):
                self._best_ma = ma
                if saved_with_torch:
                    shutil.copy2(self.last_ckpt, self.best_ckpt)
                else:
                    best_dir = self.dir / "best_model"
                    if best_dir.exists():
                        shutil.rmtree(best_dir)
                    shutil.copytree(self.dir / "last_model", best_dir)
        return snapshot

    def save_plot(self) -> None:
        try:
            import pandas as pd
            import matplotlib.pyplot as plt

            df = pd.read_csv(self.csv_path)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(df["episode"], df["reward"], label="reward")
            ma_col = [c for c in df.columns if c.startswith("ma_reward_")]
            if ma_col:
                ax.plot(df["episode"], df[ma_col[0]], label=ma_col[0])
            ax.set_xlabel("Episode")
            ax.set_ylabel("Reward")
            ax.legend()
            fig.tight_layout()
            fig.savefig(self.dir / "plot.png")
            plt.close(fig)
        except Exception:
            pass
