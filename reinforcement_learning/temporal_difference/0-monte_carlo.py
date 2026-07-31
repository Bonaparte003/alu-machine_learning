#!/usr/bin/env python3
"""Monte Carlo"""


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1,
                gamma=0.99):
    """
    performs the Monte Carlo algorithm

    Args:
        env: the openAI environment instance
        V: a numpy.ndarray of shape (s,) containing the value estimate
        policy: a function that takes in a state and returns the next action
            to take
        episodes: the total number of episodes to train over
        max_steps: the maximum number of steps per episode
        alpha: the learning rate
        gamma: the discount rate

    Returns:
        V, the updated value estimate
    """
    for _ in range(episodes):
        state = env.reset()
        episode = []

        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, reward))
            if done:
                break
            state = next_state

        G = 0
        for state, reward in reversed(episode):
            G = gamma * G + reward
            V[state] = V[state] + alpha * (G - V[state])

    return V
