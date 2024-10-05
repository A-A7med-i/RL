import numpy as np


def q_learning(
    env, num_episodes=1000, exploration_rate=0.2, learning_rate=0.2, discount_factor=0.9
) -> np.array:
    """
    Implements the Q-learning algorithm for reinforcement learning.

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

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0

        for _ in range(100):
            if np.random.uniform() < exploration_rate:
                action = np.random.randint(0, env.action_space.n)
            else:
                action = np.argmax(q_values[state, :])

            next_state, reward, done, _ = env.step(action)

            q_values[state, action] = q_values[state, action] + learning_rate * (
                (reward + discount_factor * np.max(q_values[next_state, :]))
                - q_values[state, action]
            )
            total_reward += reward

            state = next_state

            if done:
                break

    return q_values
