#!/usr/bin/env python3
"""SARSA(λ)"""
import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    performs the SARSA(λ) algorithm

    Args:
        env: the openAI environment instance
        Q: a numpy.ndarray of shape (s,a) containing the Q table
        lambtha: the eligibility trace factor
        episodes: the total number of episodes to train over
        max_steps: the maximum number of steps per episode
        alpha: the learning rate
        gamma: the discount rate
        epsilon: the initial threshold for epsilon greedy
        min_epsilon: the minimum value that epsilon should decay to
        epsilon_decay: the decay rate for updating epsilon between episodes

    Returns:
        Q, the updated Q table
    """
    n_actions = Q.shape[1]
    epsilon_init = epsilon

    def epsilon_greedy(state, epsilon):
        if np.random.uniform(0, 1) > epsilon:
            return np.argmax(Q[state])
        return np.random.randint(0, n_actions)

    for episode in range(episodes):
        state = env.reset()
        action = epsilon_greedy(state, epsilon)
        eligibility = np.zeros_like(Q)

        for _ in range(max_steps):
            next_state, reward, done, _ = env.step(action)
            next_action = epsilon_greedy(next_state, epsilon)

            delta = (reward + gamma * Q[next_state, next_action]
                     - Q[state, action])
            eligibility *= lambtha * gamma
            eligibility[state, action] += 1
            Q += alpha * delta * eligibility

            if done:
                break
            state = next_state
            action = next_action

        epsilon = (min_epsilon + (epsilon_init - min_epsilon)
                   * np.exp(-epsilon_decay * episode))

    return Q
