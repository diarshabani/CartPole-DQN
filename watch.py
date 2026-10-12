"""Watch the agent play in a window to visually interpret the results of the runs.

Usage:
  python watch.py                   # watch the best saved agent for 5 attempts
  python watch.py --progress        # replay every training checkpoint in order,
                                    # one attempt each, then the best saved agent
  python watch.py path/to/model.pt  # a specific saved model

When an attempt fails, the simulation continues briefly to show the pole fall
or the cart leave the track. Gymnasium normally ends the attempt when the pole
tilts past 12 degrees or the cart leaves the allowed area. These extra frames
do not count toward the score. Unless a model path is provided, the newest
outputs/run_* directory is used.
Press Ctrl+C at any point to stop early.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import gymnasium as gym
from gymnasium import logger as gym_logger

from config import ENV_NAME, OUTPUT_DIR
from dqn_agent import DQNAgent

# Hide Gymnasium's warning about continuing the simulation after failure
# while showing the fall animation.
gym_logger.min_level = gym_logger.ERROR

FALL_ANIMATION_FRAMES = 45  # extra game frames shown after a failure
PAUSE_SECONDS = 1.2         # pause on the final frame to view the result


def latest_run_dir() -> Path:
    run_dirs = sorted(OUTPUT_DIR.glob("run_*"))
    if not run_dirs:
        raise FileNotFoundError("No outputs/run_* directory found. Run train.py first.")
    return run_dirs[-1]


def find_latest_model() -> Path:
    model_path = latest_run_dir() / "dqn_cartpole.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"No model file at {model_path}. Run train.py first.")
    return model_path


def progress_checkpoints() -> list[Path]:
    """Find the models saved during training, in attempt order."""
    run_dir = latest_run_dir()
    checkpoints = sorted((run_dir / "checkpoints").glob("episode_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoints found in {run_dir / 'checkpoints'}. "
            "Retrain with python train.py to generate them."
        )
    return checkpoints


def animate_fall(env: gym.Env, agent: DQNAgent, state) -> None:
    """Continue the simulation briefly after failure so the fall is visible.

    Use the underlying environment to animate the fall after the attempt ends.
    """
    sim = env.unwrapped
    for _ in range(FALL_ANIMATION_FRAMES):
        action = agent.select_action(state, greedy=True)
        state, _, _, _, _ = sim.step(action)  # update the game window
        cart_position, _, pole_angle, _ = state
        if abs(cart_position) > 3.6 or abs(pole_angle) > 1.5:
            break  # stop when the cart is off-screen or the pole is nearly flat


def play_episode(env: gym.Env, agent: DQNAgent, label: str, seed: int = 1) -> float:
    """Play one attempt without random exploration, show any fall, and pause."""
    state, _ = env.reset(seed=seed)
    total_reward = 0.0
    while True:
        action = agent.select_action(state, greedy=True)
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated:
            animate_fall(env, agent, state)
            print(f"{label} | fell after {total_reward:.0f} steps")
            break
        if truncated:
            print(f"{label} | survived all {total_reward:.0f} steps (the maximum)")
            break
    time.sleep(PAUSE_SECONDS)
    return total_reward


def main() -> None:
    env = gym.make(ENV_NAME, render_mode="human")
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)

    try:
        if "--progress" in sys.argv:
            # Show how the agent played at each saved stage of training,
            # then watch the best saved model.
            for checkpoint in progress_checkpoints():
                episode_num = int(checkpoint.stem.split("_")[1])
                agent.load(checkpoint)
                play_episode(env, agent, f"after episode {episode_num:4d}")
            agent.load(find_latest_model())
            play_episode(env, agent, "final best model ")
        else:
            args = [a for a in sys.argv[1:] if not a.startswith("-")]
            model_path = Path(args[0]) if args else find_latest_model()
            print(f"Loading model: {model_path}")
            agent.load(model_path)
            for episode in range(1, 6):
                play_episode(env, agent, "trained agent", seed=episode)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
