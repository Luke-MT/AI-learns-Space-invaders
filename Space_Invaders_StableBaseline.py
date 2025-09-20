import gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
import numpy as np
import os
import time
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack


class TensorBoardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self):
        for info in self.locals['infos']:
            if 'episode' in info.keys():
                self.episode_rewards.append(info['episode']['r'])
                self.episode_lengths.append(info['episode']['l'])
                self.logger.record('rollout/ep_rew_mean', np.mean(self.episode_rewards[-100:]))
                self.logger.record('rollout/ep_len_mean', np.mean(self.episode_lengths[-100:]))
        return True


MODEL_NAME = "dqn_space_invaders"
MODEL_PATH = f"./models/{MODEL_NAME}"
LOG_PATH = f"./logs/{MODEL_NAME}"
CHECKPOINT_DIR = f"./models/checkpoints/{MODEL_NAME}"

os.makedirs("./models", exist_ok=True)
os.makedirs("./logs", exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Create training environment
env = make_atari_env('SpaceInvadersNoFrameskip-v4', n_envs=4, seed=0)
env = VecFrameStack(env, n_stack=4)

# Create a separate evaluation environment with the same specifications
eval_env = make_atari_env('SpaceInvadersNoFrameskip-v4', n_envs=1, seed=0)
eval_env = VecFrameStack(eval_env, n_stack=4)

model = DQN(
    "CnnPolicy",
    env,
    learning_rate=1e-4,
    buffer_size=100000,
    learning_starts=50000,
    batch_size=32,
    gamma=0.99,
    exploration_fraction=0.1,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,
    train_freq=4,
    gradient_steps=1,
    target_update_interval=10000,
    verbose=1,
    tensorboard_log=LOG_PATH
)


def train_model(total_timesteps=1000000):
    print("Starting training...")

    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=CHECKPOINT_DIR,
        name_prefix="dqn_space_invaders"
    )

    tensorboard_callback = TensorBoardCallback()

    # Use our explicitly created eval_env instead of letting EvalCallback create one
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{CHECKPOINT_DIR}/best_model",
        log_path=LOG_PATH,
        eval_freq=10000,
        deterministic=True,
        render=False
    )

    callbacks = [checkpoint_callback, eval_callback]

    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True,
        callback=callbacks
    )

    model.save(MODEL_PATH)
    print(f"Training completed. Final model saved to {MODEL_PATH}")


def evaluate_model(n_eval_episodes=10):
    # Use the same eval_env we created earlier
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=n_eval_episodes, deterministic=True)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    return mean_reward, std_reward


def play_game(episodes=5):
    env = gym.make('SpaceInvadersNoFrameskip-v4', render_mode='human')
    env = ResizeObservation(env, shape=84)
    env = GrayScaleObservation(env)
    env = FrameStack(env, num_stack=4)

    for episode in range(episodes):
        obs = env.reset()[0]
        done = False
        total_reward = 0

        while not done:
            obs_array = np.array(obs)
            obs_expanded = np.expand_dims(obs_array, axis=0)
            action, _ = model.predict(obs_expanded, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            total_reward += reward
            time.sleep(0.016)

        print(f"Episode {episode + 1} finished with reward {total_reward}")

    env.close()


if __name__ == "__main__":
    train_model(total_timesteps=40000000)
    evaluate_model()
    play_game()
    env.close()