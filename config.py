from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"

ENV_NAME = "CartPole-v1"
RANDOM_SEED = 42

# The model takes the four environment values through two hidden layers
# and gives a score for pushing the cart left and pushing the cart right.
MODEL_CONFIG = {
    "hidden_units": (128, 128),
}

# Training settings used by the agent and training loop.
TRAINING_CONFIG = {
    "max_episodes": 800,
    "gamma": 0.99,                 # how much future rewards count
    "learning_rate": 0.0005,
    "batch_size": 128,
    "replay_buffer_size": 10_000,  # maximum number of recorded steps kept in memory
    "learning_starts": 1_000,      # recorded steps needed before learning starts
    "target_tau": 0.005,           # how far target weights move toward trained weights each learning step
    "epsilon_start": 1.0,          # starting chance of making a random move
    "epsilon_min": 0.01,
    "epsilon_decay": 0.985,        # multiply the chance of a random move by this after each attempt
    "gradient_clip_norm": 10.0,
    "solve_window": 100,           # recent attempts used to calculate average rewards
    "checkpoint_interval": 10,     # attempts between saved models for watch.py --progress
}

EVAL_CONFIG = {
    "episodes": 20,
    "frame_capture_steps": (1, 100, 250, 450),
    "random_frame_capture_steps": (1, 5, 10, 15),
}
