import gym
from sb3_contrib import RecurrentPPO
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

MODEL_NAME = "recurrent_ppo_space_invaders"
MODEL_PATH = f"./models/{MODEL_NAME}"
LOG_PATH = f"./logs/{MODEL_NAME}"
CHECKPOINT_DIR = f"./models/checkpoints/{MODEL_NAME}"

os.makedirs("./models", exist_ok=True)
os.makedirs("./logs", exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Create training environment
env = make_atari_env('SpaceInvadersNoFrameskip-v4', n_envs=4, seed=0)
env = VecFrameStack(env, n_stack=1)  # Reduced stack size since we're using RNN

# Create a separate evaluation environment
eval_env = make_atari_env('SpaceInvadersNoFrameskip-v4', n_envs=1, seed=0)
eval_env = VecFrameStack(eval_env, n_stack=1)

# Create RecurrentPPO model with LSTM
model = RecurrentPPO(
    "CnnLstmPolicy",
    env,
    learning_rate=1e-4,
    n_steps=128,
    batch_size=256,
    n_epochs=4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    clip_range_vf=None,
    normalize_advantage=True,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    use_sde=False,
    sde_sample_freq=-1,
    #lstm_hidden_size=256,  # Size of LSTM hidden states
    #n_lstm_layers=1,       # Number of LSTM layers
    verbose=1,
    tensorboard_log=LOG_PATH
)

def train_model(total_timesteps=10000000):
    print("Starting training...")

    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=CHECKPOINT_DIR,
        name_prefix="recurrent_ppo_space_invaders"
    )

    tensorboard_callback = TensorBoardCallback()

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
    # For RecurrentPPO, we need to handle LSTM states
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=n_eval_episodes,
        deterministic=True
    )
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    return mean_reward, std_reward

def play_game(episodes=5):
    env = gym.make('SpaceInvadersNoFrameskip-v4', render_mode='human')
    env = ResizeObservation(env, shape=84)
    env = GrayScaleObservation(env)
    env = FrameStack(env, num_stack=1)  # Reduced stack size since we're using RNN

    for episode in range(episodes):
        obs = env.reset()[0]
        done = False
        total_reward = 0
        # Initialize LSTM states
        lstm_states = None

        while not done:
            obs_array = np.array(obs)
            obs_expanded = np.expand_dims(obs_array, axis=0)
            # Get action and update LSTM states
            action, lstm_states = model.predict(
                obs_expanded,
                state=lstm_states,
                deterministic=True
            )
            obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            total_reward += reward
            time.sleep(0.016)

        print(f"Episode {episode + 1} finished with reward {total_reward}")

    env.close()

if __name__ == "__main__":
    train_model(total_timesteps=4000000)
    evaluate_model()
    play_game()
    env.close()