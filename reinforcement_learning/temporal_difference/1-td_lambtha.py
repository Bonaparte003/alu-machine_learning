#!/usr/bin/env python3
"""TD(λ)"""
import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100,
               alpha=0.1, gamma=0.99):
    """
    performs the TD(λ) algorithm

    Args:
        env: the openAI environment instance
        V: a numpy.ndarray of shape (s,) containing the value estimate
        policy: a function that takes in a state and returns the next action
            to take
        lambtha: the eligibility trace factor
        episodes: the total number of episodes to train over
        max_steps: the maximum number of steps per episode
        alpha: the learning rate
        gamma: the discount rate

    Returns:
        V, the updated value estimate
    """
    for _ in range(episodes):
        state = env.reset()
        eligibility = np.zeros_like(V)

        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, done, _ = env.step(action)

            delta = reward + gamma * V[next_state] - V[state]
            eligibility *= lambtha * gamma
            eligibility[state] += 1
            V = V + alpha * delta * eligibility

            if done:
                break
            state = next_state

    return V
