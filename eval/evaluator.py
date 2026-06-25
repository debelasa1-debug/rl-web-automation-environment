"""
eval/evaluator.py
Evaluation system for the RL Web Automation Environment.

Tracks per-episode and aggregate metrics, writes results to JSON/CSV,
and produces a human-readable summary report.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Optional

from loguru import logger


# ---------------------------------------------------------------------------
# Episode record
# ---------------------------------------------------------------------------

@dataclass
class EpisodeRecord:
    episode_id: int
    task_id: str
    task_category: str
    task_difficulty: str
    agent_type: str
    success: bool
    total_reward: float
    steps: int
    duration_seconds: float
    reward_breakdown: dict = field(default_factory=dict)
    milestones_hit: list[int] = field(default_factory=list)
    final_url: str = ""
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

@dataclass
class AggregateMetrics:
    n_episodes: int = 0
    success_rate: float = 0.0
    avg_reward: float = 0.0
    avg_steps: float = 0.0
    avg_duration: float = 0.0
    reward_std: float = 0.0
    steps_std: float = 0.0
    by_task: dict[str, dict] = field(default_factory=dict)
    by_category: dict[str, dict] = field(default_factory=dict)
    by_difficulty: dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "n_episodes": self.n_episodes,
            "success_rate": round(self.success_rate, 4),
            "avg_reward": round(self.avg_reward, 3),
            "avg_steps": round(self.avg_steps, 2),
            "avg_duration_s": round(self.avg_duration, 2),
            "reward_std": round(self.reward_std, 3),
            "steps_std": round(self.steps_std, 2),
            "by_task": self.by_task,
            "by_category": self.by_category,
            "by_difficulty": self.by_difficulty,
        }


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    """
    Records, analyses, and exports RL episode metrics.
    """

    def __init__(
        self,
        output_dir: Path = Path("logs"),
        agent_type: str = "unknown",
    ):
        self.output_dir = output_dir
        self.agent_type = agent_type
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._episodes: list[EpisodeRecord] = []
        self._current_start: Optional[float] = None
        self._episode_id: int = 0
        logger.info("Evaluator initialised → output={}", output_dir)

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def start_episode(self) -> None:
        self._current_start = time.time()

    def record_episode(
        self,
        task,                        # Task object
        success: bool,
        total_reward: float,
        steps: int,
        reward_breakdown: Optional[dict] = None,
        milestones_hit: Optional[list[int]] = None,
        final_url: str = "",
        notes: str = "",
    ) -> EpisodeRecord:
        duration = time.time() - (self._current_start or time.time())
        self._episode_id += 1

        record = EpisodeRecord(
            episode_id=self._episode_id,
            task_id=task.task_id,
            task_category=task.category.value,
            task_difficulty=task.difficulty.value,
            agent_type=self.agent_type,
            success=success,
            total_reward=round(total_reward, 3),
            steps=steps,
            duration_seconds=round(duration, 2),
            reward_breakdown=reward_breakdown or {},
            milestones_hit=milestones_hit or [],
            final_url=final_url,
            notes=notes,
        )
        self._episodes.append(record)

        icon = "✅" if success else "❌"
        logger.info(
            "{} Episode {:3d} | task={:<15} | reward={:+7.2f} | steps={:3d} | {:.1f}s",
            icon, self._episode_id, task.task_id, total_reward, steps, duration,
        )
        return record

    # ------------------------------------------------------------------
    # Metrics computation
    # ------------------------------------------------------------------

    def compute_metrics(self) -> AggregateMetrics:
        if not self._episodes:
            return AggregateMetrics()

        rewards = [e.total_reward for e in self._episodes]
        steps_list = [e.steps for e in self._episodes]
        durations = [e.duration_seconds for e in self._episodes]

        metrics = AggregateMetrics(
            n_episodes=len(self._episodes),
            success_rate=mean(1.0 if e.success else 0.0 for e in self._episodes),
            avg_reward=mean(rewards),
            avg_steps=mean(steps_list),
            avg_duration=mean(durations),
            reward_std=stdev(rewards) if len(rewards) > 1 else 0.0,
            steps_std=stdev(steps_list) if len(steps_list) > 1 else 0.0,
        )

        # Per-task breakdown
        for e in self._episodes:
            if e.task_id not in metrics.by_task:
                metrics.by_task[e.task_id] = {"n": 0, "success": 0, "rewards": [], "steps": []}
            d = metrics.by_task[e.task_id]
            d["n"] += 1
            d["success"] += int(e.success)
            d["rewards"].append(e.total_reward)
            d["steps"].append(e.steps)

        for tid, d in metrics.by_task.items():
            d["success_rate"] = round(d["success"] / d["n"], 3)
            d["avg_reward"] = round(mean(d["rewards"]), 3)
            d["avg_steps"] = round(mean(d["steps"]), 2)
            del d["rewards"], d["steps"]

        # Per-category breakdown
        for e in self._episodes:
            cat = e.task_category
            if cat not in metrics.by_category:
                metrics.by_category[cat] = {"n": 0, "success": 0, "total_reward": 0.0}
            metrics.by_category[cat]["n"] += 1
            metrics.by_category[cat]["success"] += int(e.success)
            metrics.by_category[cat]["total_reward"] += e.total_reward
        for cat, d in metrics.by_category.items():
            d["success_rate"] = round(d["success"] / d["n"], 3)
            d["avg_reward"] = round(d["total_reward"] / d["n"], 3)
            del d["total_reward"]

        # Per-difficulty breakdown
        for e in self._episodes:
            diff = e.task_difficulty
            if diff not in metrics.by_difficulty:
                metrics.by_difficulty[diff] = {"n": 0, "success": 0}
            metrics.by_difficulty[diff]["n"] += 1
            metrics.by_difficulty[diff]["success"] += int(e.success)
        for diff, d in metrics.by_difficulty.items():
            d["success_rate"] = round(d["success"] / d["n"], 3)

        return metrics

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def save_json(self, filename: str = "results.json") -> Path:
        path = self.output_dir / filename
        metrics = self.compute_metrics()
        data = {
            "summary": metrics.to_dict(),
            "episodes": [e.to_dict() for e in self._episodes],
        }
        path.write_text(json.dumps(data, indent=2))
        logger.info("Results saved → {}", path)
        return path

    def save_csv(self, filename: str = "episodes.csv") -> Path:
        path = self.output_dir / filename
        if not self._episodes:
            logger.warning("No episodes to export")
            return path
        fieldnames = list(self._episodes[0].to_dict().keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for ep in self._episodes:
                row = ep.to_dict()
                row["milestones_hit"] = str(row["milestones_hit"])
                row["reward_breakdown"] = str(row["reward_breakdown"])
                writer.writerow(row)
        logger.info("CSV saved → {}", path)
        return path

    def print_summary(self) -> None:
        metrics = self.compute_metrics()
        m = metrics.to_dict()
        lines = [
            "",
            "╔══════════════════════════════════════════════════════╗",
            "║         RL WEB AUTOMATION — EVALUATION SUMMARY       ║",
            "╚══════════════════════════════════════════════════════╝",
            f"  Agent        : {self.agent_type}",
            f"  Episodes     : {m['n_episodes']}",
            f"  Success Rate : {m['success_rate']:.1%}",
            f"  Avg Reward   : {m['avg_reward']:+.2f}  (σ={m['reward_std']:.2f})",
            f"  Avg Steps    : {m['avg_steps']:.1f}     (σ={m['steps_std']:.2f})",
            f"  Avg Duration : {m['avg_duration_s']:.1f}s",
            "",
            "  ── By Category ──",
        ]
        for cat, d in sorted(m["by_category"].items()):
            lines.append(
                f"  {cat:<20}: success={d['success_rate']:.0%}  "
                f"avg_reward={d['avg_reward']:+.1f}  n={d['n']}"
            )
        lines += ["", "  ── By Difficulty ──"]
        for diff, d in sorted(m["by_difficulty"].items()):
            lines.append(f"  {diff:<12}: success={d['success_rate']:.0%}  n={d['n']}")
        lines += ["", "  ── Per Task ──"]
        for tid, d in sorted(m["by_task"].items()):
            lines.append(
                f"  {tid:<15}: success={d['success_rate']:.0%}  "
                f"avg_reward={d['avg_reward']:+.1f}  avg_steps={d['avg_steps']:.0f}  n={d['n']}"
            )
        lines.append("")
        print("\n".join(lines))
