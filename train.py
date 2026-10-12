"""Runs the training, tests the agent, and saves the results.

Run with:  python train.py
"""
from __future__ import annotations

import random
import time

import gymnasium as gym
import numpy as np
import torch
from gymnasium import logger as gym_logger

from artifacts import (
    create_run_dir,
    save_eval_chart,
    save_eval_csv,
    save_frame_strip,
    save_metrics_csv,
    save_run_summary,
    save_training_charts,
)
from config import ENV_NAME, EVAL_CONFIG, RANDOM_SEED, TRAINING_CONFIG
from dqn_agent import DQNAgent

FALL_ANIMATION_FRAMES = 45  # extra frames to show the result of a failed attempt


def set_global_seeds(seed: int) -> None:
    """Set the random seeds so we can repeat the training with the same settings."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_agent(env: gym.Env, agent: DQNAgent, model_path) -> tuple[list[dict], int | None, float]:
    """Train the agent and test how well it keeps the pole balanced.

    Every 25 attempts, check whether the recent average score is at least 200.
    If it is, test the agent over 10 attempts with random moves turned off.
    Training stops when the average test score reaches the environment's
    target score or we reach the maximum number of training attempts.

    Save the model when its recent average score improves and when it passes
    the test.
    """
    solve_threshold = env.spec.reward_threshold  # target score set by the environment
    window = TRAINING_CONFIG["solve_window"]
    check_env = gym.make(ENV_NAME)
    episode_metrics: list[dict] = []
    recent_rewards: list[float] = []
    solved_episode: int | None = None
    best_rolling_mean = float("-inf")

    # Save models during training so watch.py --progress can show earlier stages.
    checkpoint_dir = model_path.parent / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    for episode in range(1, TRAINING_CONFIG["max_episodes"] + 1):
        state, _ = env.reset(seed=RANDOM_SEED + episode)
        episode_reward = 0.0
        episode_losses: list[float] = []
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Keep a recorded history of each move and its result.
            # Reaching the step limit ends the attempt but does not count as a failure.
            agent.store_transition(state, action, reward, next_state, terminated)
            loss = agent.learn()
            if loss is not None:
                episode_losses.append(loss)

            state = next_state
            episode_reward += reward

        agent.decay_epsilon()
        recent_rewards = (recent_rewards + [episode_reward])[-window:]
        rolling_mean = float(np.mean(recent_rewards))

        # Save a summary for this attempt; the history of each step stays in memory.
        episode_metrics.append(
            {
                "episode": episode,
                "reward": episode_reward,
                "rolling_mean_reward": round(rolling_mean, 2),
                "epsilon": round(agent.epsilon, 4),
                "mean_loss": round(float(np.mean(episode_losses)), 6) if episode_losses else None,
            }
        )

        if episode % 25 == 0:
            print(
                f"Episode {episode:4d} | reward {episode_reward:6.1f} | "
                f"{window}-ep avg {rolling_mean:6.1f} | epsilon {agent.epsilon:.3f}"
            )

        # Save the first model and then one at each checkpoint interval.
        if episode == 1 or episode % TRAINING_CONFIG["checkpoint_interval"] == 0:
            agent.save(checkpoint_dir / f"episode_{episode:04d}.pt")

        # Keep the model when its recent average score improves.
        if len(recent_rewards) == window and rolling_mean > best_rolling_mean:
            best_rolling_mean = rolling_mean
            agent.save(model_path)

        # Test 10 attempts with random moves turned off.
        if episode % 25 == 0 and rolling_mean >= 200:
            greedy_mean = float(np.mean(evaluate_agent(check_env, agent, episodes=10)))
            print(f"          | greedy evaluation over 10 episodes: {greedy_mean:.1f}")
            if greedy_mean >= solve_threshold:
                solved_episode = episode
                agent.save(model_path)  # save the model that reached the target score
                print(f"Solved at episode {episode}: greedy 10-episode average "
                      f"{greedy_mean:.1f} >= {solve_threshold}")
                break

    check_env.close()
    if best_rolling_mean == float("-inf"):
        # Save a model even if there were too few attempts for the full average.
        agent.save(model_path)

    return episode_metrics, solved_episode, best_rolling_mean


def evaluate_agent(env: gym.Env, agent: DQNAgent, episodes: int) -> list[float]:
    """Test the model with random moves turned off and record each attempt's score."""
    rewards = []
    for episode in range(episodes):
        state, _ = env.reset(seed=10_000 + episode)
        episode_reward = 0.0
        done = False
        while not done:
            action = agent.select_action(state, greedy=True)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        rewards.append(episode_reward)
    return rewards


def capture_fallen_frame(env: gym.Env, agent: DQNAgent | None, state) -> np.ndarray:
    """Keep rendering after a failed attempt so the saved image shows the result.

    Gymnasium ends an attempt as soon as the pole tips past 12 degrees or the
    cart moves out of bounds. A few extra frames let us see the pole fall or
    the cart leave the track. These extra steps are only used for the image.
    """
    prev_level = gym_logger.min_level
    gym_logger.min_level = gym_logger.ERROR  # hide warnings while showing the fall
    sim = env.unwrapped
    for _ in range(FALL_ANIMATION_FRAMES):
        action = (
            env.action_space.sample()
            if agent is None
            else agent.select_action(state, greedy=True)
        )
        state, _, _, _, _ = sim.step(action)
        cart_position, _, pole_angle, _ = state
        if abs(cart_position) > 3.6 or abs(pole_angle) > 1.5:
            break  # stop once the cart is off-screen or the pole is almost flat
    gym_logger.min_level = prev_level
    return env.render()


def capture_frames(agent: DQNAgent | None, capture_steps: tuple[int, ...], seed: int) -> dict[int, np.ndarray]:
    """Capture game images at selected steps during one attempt.

    When no agent is provided, use random left or right moves to show the
    starting performance before training.
    """
    env = gym.make(ENV_NAME, render_mode="rgb_array")
    env.action_space.seed(seed)  # use the same random moves when this attempt is repeated
    state, _ = env.reset(seed=seed)
    frames: dict[int, np.ndarray] = {}
    max_step = max(capture_steps)

    for step in range(1, max_step + 1):
        if step in capture_steps:
            frames[step] = env.render()
        action = (
            env.action_space.sample()
            if agent is None
            else agent.select_action(state, greedy=True)
        )
        state, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            # Show the fall after a failure, or capture the final frame at the step limit.
            frames[step + 1] = (
                capture_fallen_frame(env, agent, state) if terminated else env.render()
            )
            break

    env.close()
    return frames


def main() -> None:
    set_global_seeds(RANDOM_SEED)
    run_dir = create_run_dir()

    env = gym.make(ENV_NAME)
    env.action_space.seed(RANDOM_SEED)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    solve_threshold = env.spec.reward_threshold

    agent = DQNAgent(state_size, action_size)

    # Capture the agent's starting performance using random left or right moves.
    random_frames = capture_frames(None, EVAL_CONFIG["random_frame_capture_steps"], seed=7)

    model_path = run_dir / "dqn_cartpole.pt"
    start_time = time.time()
    episode_metrics, solved_episode, best_rolling_mean = train_agent(env, agent, model_path)
    training_minutes = (time.time() - start_time) / 60
    env.close()

    # Load the saved model for the final tests and game images.
    agent.load(model_path)
    eval_env = gym.make(ENV_NAME)
    eval_rewards = evaluate_agent(eval_env, agent, EVAL_CONFIG["episodes"])
    eval_env.close()

    trained_frames = capture_frames(agent, EVAL_CONFIG["frame_capture_steps"], seed=7)

    save_metrics_csv(run_dir, episode_metrics)
    save_eval_csv(run_dir, eval_rewards)
    save_training_charts(run_dir, episode_metrics, solve_threshold)
    save_eval_chart(run_dir, eval_rewards)
    save_frame_strip(run_dir, random_frames, "agent_untrained.png",
                     "Untrained Agent (random actions) - the pole falls within a few steps")
    save_frame_strip(run_dir, trained_frames, "agent_trained.png",
                     "Trained DQN Agent - the pole stays balanced")

    summary = [
        f"Environment: {ENV_NAME}",
        f"Episodes trained: {len(episode_metrics)}",
        f"Solved (greedy 10-episode average >= threshold) at episode: "
        f"{solved_episode if solved_episode else 'not solved'}",
        f"Best {TRAINING_CONFIG['solve_window']}-episode average: {best_rolling_mean:.2f}",
        f"Final {TRAINING_CONFIG['solve_window']}-episode average: "
        f"{episode_metrics[-1]['rolling_mean_reward']}",
        f"Greedy evaluation over {EVAL_CONFIG['episodes']} episodes: "
        f"mean {np.mean(eval_rewards):.1f}, min {np.min(eval_rewards):.0f}, max {np.max(eval_rewards):.0f}",
        f"Training time: {training_minutes:.1f} minutes",
        f"Replay buffer size at end: {len(agent.replay_buffer)}",
    ]
    save_run_summary(run_dir, summary)

    print("")
    for line in summary:
        print(line)
    print(f"\nOutputs saved to: {run_dir}")


if __name__ == "__main__":
    main()
