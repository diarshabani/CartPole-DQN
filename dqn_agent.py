"""The CartPole model, its recorded history, and the code that trains it."""
from __future__ import annotations

import random
from collections import deque

import numpy as np
import torch
from torch import nn

from config import MODEL_CONFIG, TRAINING_CONFIG


class QNetwork(nn.Module):
    """Uses the four environment values to estimate a score for left and right.

    These values are cart position, cart velocity, pole angle, and pole angular
    velocity.
    """

    def __init__(self, state_size: int, action_size: int) -> None:
        super().__init__()
        hidden_one, hidden_two = MODEL_CONFIG["hidden_units"]
        self.layers = nn.Sequential(
            nn.Linear(state_size, hidden_one),
            nn.ReLU(),
            nn.Linear(hidden_one, hidden_two),
            nn.ReLU(),
            nn.Linear(hidden_two, action_size),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.layers(state)


class ReplayBuffer:
    """Keeps a recorded history of steps in memory, removing the oldest when full.

    Each recording includes the state, action, reward, next state, and failure
    flag. Random batches mix recordings from different points in the game.
    """

    def __init__(self, capacity: int) -> None:
        self.memory: deque = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done) -> None:
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        """Select random recordings and convert them into PyTorch tensors."""
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.as_tensor(np.array(states), dtype=torch.float32),
            torch.as_tensor(actions, dtype=torch.int64).unsqueeze(1),
            torch.as_tensor(rewards, dtype=torch.float32).unsqueeze(1),
            torch.as_tensor(np.array(next_states), dtype=torch.float32),
            torch.as_tensor(dones, dtype=torch.float32).unsqueeze(1),
        )

    def __len__(self) -> int:
        return len(self.memory)


class DQNAgent:
    """Chooses left or right and trains the model using recorded steps."""

    def __init__(self, state_size: int, action_size: int) -> None:
        self.action_size = action_size
        self.gamma = TRAINING_CONFIG["gamma"]
        self.batch_size = TRAINING_CONFIG["batch_size"]
        self.epsilon = TRAINING_CONFIG["epsilon_start"]

        # The online network learns from the recordings. The target network
        # follows it gradually to help keep learning stable.
        self.online_network = QNetwork(state_size, action_size)
        self.target_network = QNetwork(state_size, action_size)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(
            self.online_network.parameters(), lr=TRAINING_CONFIG["learning_rate"]
        )
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss limits the effect of large prediction errors
        self.replay_buffer = ReplayBuffer(TRAINING_CONFIG["replay_buffer_size"])
        self.learn_steps = 0

    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        """Try a random move with chance epsilon; otherwise use the higher score.

        Greedy evaluation turns random moves off.
        """
        if not greedy and random.random() < self.epsilon:
            return random.randrange(self.action_size)

        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = self.online_network(state_tensor)
        return int(q_values.argmax(dim=1).item())

    def store_transition(self, state, action, reward, next_state, done) -> None:
        self.replay_buffer.push(state, action, reward, next_state, done)

    def decay_epsilon(self) -> None:
        """Reduce the chance of random moves after each attempt, down to a minimum."""
        self.epsilon = max(
            TRAINING_CONFIG["epsilon_min"],
            self.epsilon * TRAINING_CONFIG["epsilon_decay"],
        )

    def learn(self) -> float | None:
        """Train on one random batch of recordings and return the learning loss."""
        if len(self.replay_buffer) < max(self.batch_size, TRAINING_CONFIG["learning_starts"]):
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Get the model's estimated score for each recorded action.
        predicted_q = self.online_network(states).gather(1, actions)

        # With Double DQN, the online network picks the next action and the
        # target network estimates its score. This helps avoid overestimating it.
        with torch.no_grad():
            next_actions = self.online_network(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_network(next_states).gather(1, next_actions)
            target_q = rewards + self.gamma * next_q * (1.0 - dones)

        loss = self.loss_fn(predicted_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.online_network.parameters(), TRAINING_CONFIG["gradient_clip_norm"]
        )
        self.optimizer.step()

        # Move the target weights a little closer to the trained weights
        # after each learning step.
        self.learn_steps += 1
        tau = TRAINING_CONFIG["target_tau"]
        with torch.no_grad():
            for target_param, online_param in zip(
                self.target_network.parameters(), self.online_network.parameters()
            ):
                target_param.mul_(1.0 - tau).add_(online_param, alpha=tau)

        return float(loss.item())

    def save(self, path) -> None:
        torch.save(self.online_network.state_dict(), path)

    def load(self, path) -> None:
        state_dict = torch.load(path, map_location="cpu")
        self.online_network.load_state_dict(state_dict)
        self.target_network.load_state_dict(state_dict)
