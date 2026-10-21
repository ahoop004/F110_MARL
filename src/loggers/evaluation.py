"""W&B metrics for frozen-policy evaluation, separate from training curves."""
from __future__ import annotations

import math

from loggers.wandb_logger import WandbLogger


def scalar_metrics(summary, namespace):
    return {f"{namespace}/{key}": value for key, value in summary.items()
            if isinstance(value, (int, float)) and math.isfinite(value)}


class EvaluationWandbLogger:
    def __init__(self, scenario, run_id, checkpoint, checkpoint_hash, checkpoint_steps,
                 actor_evaluation, protocol, output_dir):
        cfg = scenario.get("wandb", {})
        config = {**scenario, "standalone_evaluation": dict(
            checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash,
            checkpoint_environment_steps=checkpoint_steps, actor=actor_evaluation,
            protocol=protocol, output_dir=str(output_dir), run_id=run_id)}
        self.logger = WandbLogger(
            project=cfg.get("project", "f110"), entity=cfg.get("entity"), config=config,
            name=cfg.get("name") or run_id, tags=cfg.get("tags", []), group=cfg.get("group"),
            job_type=cfg.get("job_type", "evaluation"), notes=cfg.get("notes"), mode=cfg.get("mode", "online"),
            logging_config=cfg.get("logging"), run_id=run_id)
        self.completed_episodes = 0

    @property
    def url(self):
        return self.logger.wandb_url

    def log_episode(self, summary, episode, seed):
        metrics = scalar_metrics(summary, "eval_episode")
        metrics.update({"eval_episode/number": episode, "eval_episode/seed": seed})
        self.logger.log_metrics(metrics)
        self.completed_episodes = episode

    def finish(self, summary=None):
        try:
            if summary is not None:
                metrics = scalar_metrics(summary, "eval")
                # All comparison runs end at the same episode count. The source
                # checkpoint's training step stays in config, not on the eval axis.
                metrics["eval/environment_steps"] = 0
                self.logger.log_metrics(metrics)
                if self.logger.run is not None:
                    self.logger.run.summary.update(self.logger._filter_metrics(metrics))
                    self.logger.run.summary["evaluation/completed_episodes"] = self.completed_episodes
        finally:
            self.logger.finish()
