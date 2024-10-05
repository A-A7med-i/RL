import numpy as np


def sarsa_learning(
        env,
        num_episodes: int = 1000,
        exploration_rate: float = 0.2,
        learning_rate: float = 0.2,
        discount_factor: float = 0.9,
) -> np.ndarray:
    """
    Implements the SARSA algorithm for reinforcement learning.

    Args:
        env: The environment to learn in.
        num_episodes: The number of episodes to run.
        exploration_rate: The probability of choosing a random action.
        learning_rate: The learning rate for updating the Q-values.
        discount_factor: The discount factor for future rewards.

    Returns:
        The learned Q-values.
    """
    q_values = np.zeros([env.observation_space.n, env.action_space.n])

    for iteration in range(num_episodes):
        state = env.reset()
        total_reward = 0

        action = np.random.randint(0, env.action_space.n)

        for step in range(99):
            next_state, reward, done, _ = env.step(action)

            if np.random.uniform() < exploration_rate:
                next_action = np.random.randint(0, env.action_space.n)
            else:
                next_action = np.argmax(q_values[next_state, :])

            q_values[state, action] = q_values[state, action] + learning_rate * (
                    (reward + discount_factor * q_values[next_state, next_action])
                    - q_values[state, action]
            )
            total_reward += reward

            if done:
                break

            action = next_action
            state = next_state

    return q_values
