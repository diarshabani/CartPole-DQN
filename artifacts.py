"""Save and generate the charts, CSV files, game images, and results summary."""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # generate charts without opening a window
import matplotlib.pyplot as plt
import numpy as np

from config import OUTPUT_DIR, TRAINING_CONFIG


def create_run_dir() -> Path:
    run_dir = OUTPUT_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_metrics_csv(run_dir: Path, episode_metrics: list[dict]) -> None:
    """Save the recorded training results in a CSV file."""
    path = run_dir / "training_metrics.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(episode_metrics[0].keys()))
        writer.writeheader()
        writer.writerows(episode_metrics)


def save_eval_csv(run_dir: Path, eval_rewards: list[float]) -> None:
    """Save the recorded score for each evaluation attempt."""
    path = run_dir / "evaluation_rewards.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward"])
        writer.writerows([(i + 1, r) for i, r in enumerate(eval_rewards)])


def save_training_charts(run_dir: Path, episode_metrics: list[dict], solve_threshold: float) -> None:
    """Show agent scores, exploration, and learning errors over time."""
    episodes = [m["episode"] for m in episode_metrics]
    rewards = [m["reward"] for m in episode_metrics]
    epsilons = [m["epsilon"] for m in episode_metrics]
    losses = [m["mean_loss"] for m in episode_metrics if m["mean_loss"] is not None]
    loss_episodes = [m["episode"] for m in episode_metrics if m["mean_loss"] is not None]

    window = TRAINING_CONFIG["solve_window"]
    moving_avg = [
        float(np.mean(rewards[max(0, i - window + 1) : i + 1])) for i in range(len(rewards))
    ]

    # Scores for each attempt, the recent average, and the score needed to solve.
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, rewards, alpha=0.35, label="Episode reward")
    plt.plot(episodes, moving_avg, linewidth=2, label=f"{window}-episode average")
    plt.axhline(solve_threshold, linestyle="--", color="red", label=f"Solve threshold ({solve_threshold:.0f})")
    plt.xlabel("Episode")
    plt.ylabel("Total reward (steps balanced)")
    plt.title("DQN Training Rewards on CartPole-v1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "training_rewards.png", dpi=150)
    plt.close()

    # Chance of choosing a random action during each attempt.
    plt.figure(figsize=(10, 4))
    plt.plot(episodes, epsilons, color="tab:orange")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.title("Epsilon-Greedy Exploration Decay")
    plt.tight_layout()
    plt.savefig(run_dir / "epsilon_decay.png", dpi=150)
    plt.close()

    # Average learning error from the updates during each attempt.
    plt.figure(figsize=(10, 4))
    plt.plot(loss_episodes, losses, color="tab:green")
    plt.xlabel("Episode")
    plt.ylabel("Mean TD loss")
    plt.title("Temporal-Difference Loss During Training")
    plt.tight_layout()
    plt.savefig(run_dir / "training_loss.png", dpi=150)
    plt.close()


def save_eval_chart(run_dir: Path, eval_rewards: list[float]) -> None:
    plt.figure(figsize=(10, 4))
    plt.bar(range(1, len(eval_rewards) + 1), eval_rewards, color="tab:blue")
    plt.axhline(float(np.mean(eval_rewards)), linestyle="--", color="black",
                label=f"Mean = {np.mean(eval_rewards):.1f}")
    plt.xlabel("Evaluation episode")
    plt.ylabel("Total reward")
    plt.title("Greedy-Policy Evaluation (no exploration)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "evaluation_rewards.png", dpi=150)
    plt.close()


def save_frame_strip(run_dir: Path, frames: dict[int, np.ndarray], filename: str, title: str) -> None:
    """Arrange game images in a labeled row to visually interpret the results."""
    steps = sorted(frames.keys())
    fig, axes = plt.subplots(1, len(steps), figsize=(4 * len(steps), 3.2))
    if len(steps) == 1:
        axes = [axes]
    for ax, step in zip(axes, steps):
        ax.imshow(frames[step])
        ax.set_title(f"Step {step}")
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(run_dir / filename, dpi=150)
    plt.close(fig)


def save_run_summary(run_dir: Path, summary_lines: list[str]) -> None:
    with open(run_dir / "run_summary.txt", "w") as f:
        f.write("\n".join(summary_lines) + "\n")
