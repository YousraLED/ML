
"""
Code references:
    Most of the code is inspired from https://github.com/hakantekgul/cs7641-assignment4
    pymdptoolbox mdp.py was modified for testing and plotting purposes
"""

import numpy as np
import gym
import time
import mdp, mdptoolbox, mdptoolbox.example
from pathlib import Path


from plot import plot_qlearning_experiment_results_forest_forest, plot_iteration_experiment_results_forest, \
    plot_q_learning_experiment_results, plot_iteration_experiment_results, plot_policy_map, \
    plot_q_learning_experiment_results_with_gammas, plot_qlearning_experiment_results_forest_forest_with_gammas


def perform_frozen_lake_experiments(output_path, max_tries=10):
    # 0 = left; 1 = down; 2 = right;  3 = up

    experiment_name = "Frozen Lake"
    experiment_output_path = fr"{output_path}\{experiment_name}"
    Path(experiment_output_path).mkdir(parents=True, exist_ok=True)

    environment = 'FrozenLake-v0'  # 'FrozenLake8x8-v0' #
    env = gym.make(environment)
    env = env.unwrapped
    desc = env.unwrapped.desc

    algorithm = "Policy Iteration"
    gamma_arr, iters, list_scores, time_array = perform_policy_iteration(env, max_tries)
    best_vals = []
    plot_iteration_experiment_results(experiment_output_path,
                                      algorithm,
                                      experiment_name,
                                      gamma_arr,
                                      iters,
                                      list_scores,
                                      time_array,
                                      best_vals)

    algorithm = "Value Iteration"
    best_vals, gamma_arr, iters, list_scores, time_array = perform_value_iteration(desc, env, max_tries)
    plot_iteration_experiment_results(experiment_output_path,
                                      algorithm,
                                      experiment_name,
                                      gamma_arr,
                                      iters,
                                      list_scores,
                                      time_array,
                                      best_vals)

    algorithm = "Q-Learning"
    Q_array, epsilons, averages_array, reward_array, size_array, time_array = perform_q_learning(env, environment)
    plot_q_learning_experiment_results(experiment_output_path,
                                       Q_array,
                                       algorithm,
                                       averages_array,
                                       epsilons,
                                       experiment_name,
                                       reward_array,
                                       size_array,
                                       time_array)

    algorithm = "Q-Learning (Varying Gamma)"
    Q_array, gammas, averages_array, reward_array, size_array, time_array = perform_q_learning_varying_gamma(env, environment)
    plot_q_learning_experiment_results_with_gammas(experiment_output_path,
                                                       Q_array,
                                                       algorithm,
                                                       averages_array,
                                                       gammas,
                                                       experiment_name,
                                                       reward_array,
                                                       size_array,
                                                       time_array)


def perform_forest_experiments(output_path):
    experiment_name = "Forest Management"
    experiment_output_path = fr"{output_path}\{experiment_name}"
    Path(experiment_output_path).mkdir(parents=True, exist_ok=True)

    states = 3000

    algorithm = "Policy Iteration"
    gamma_arr, iters, time_array, value_f = perform_policy_iteration_mdp(states)
    plot_iteration_experiment_results_forest(experiment_output_path,
                                             algorithm,
                                             experiment_name,
                                             gamma_arr,
                                             iters,
                                             time_array,
                                             value_f)

    algorithm = "Value Iteration"
    gamma_arr, iters, time_array, value_f = perform_value_iteration_mdp(states)
    plot_iteration_experiment_results_forest(experiment_output_path,
                                             algorithm,
                                             experiment_name,
                                             gamma_arr,
                                             iters,
                                             time_array,
                                             value_f)

    algorithm = "Q Learning"
    Q_table, rew_array, epsilons, time_array, time_array = perform_q_learning_mdp(states)
    plot_qlearning_experiment_results_forest_forest(experiment_output_path,
                                                    Q_table,
                                                    algorithm,
                                                    experiment_name,
                                                    rew_array,
                                                    epsilons,
                                                    time_array)

    algorithm = "Q-Learning (Varying Gamma)"
    Q_table, rew_array, gammas, time_array, time_array = perform_q_learning_mdp_with_varying_gamma(states)
    plot_qlearning_experiment_results_forest_forest_with_gammas(experiment_output_path,
                                                    Q_table,
                                                    algorithm,
                                                    experiment_name,
                                                    rew_array,
                                                    gammas,
                                                    time_array)


    return


def perform_q_learning(env, environment):
    ### Q-LEARNING #####
    print('Q LEARNING WITH FROZEN LAKE')
    st = time.time()
    reward_array = []
    iter_array = []
    size_array = []
    chunks_array = []
    averages_array = []
    time_array = []
    Q_array = []
    epsilons = [0.05, 0.15, 0.25, 0.5, 0.75, 0.90]
    gammas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for epsilon in epsilons:
        Q = np.zeros((env.observation_space.n, env.action_space.n))
        rewards = []
        iters = []
        optimal = [0] * env.observation_space.n
        alpha = 0.85
        gamma = 0.95
        episodes = 30000
        env = gym.make(environment)
        env = env.unwrapped
        desc = env.unwrapped.desc
        for episode in range(episodes):
            state = env.reset()
            done = False
            t_reward = 0
            max_steps = 1000000
            for i in range(max_steps):
                if done:
                    break
                current = state
                if np.random.rand() < (epsilon):
                    action = np.argmax(Q[current, :])
                else:
                    action = env.action_space.sample()

                state, reward, done, info = env.step(action)
                t_reward += reward
                Q[current, action] += alpha * (reward + gamma * np.max(Q[state, :]) - Q[current, action])
            epsilon = (1 - 2.71 ** (-episode / 1000))
            rewards.append(t_reward)
            iters.append(i)

        for k in range(env.observation_space.n):
            optimal[k] = np.argmax(Q[k, :])

        reward_array.append(rewards)
        iter_array.append(iters)
        Q_array.append(Q)

        env.close()
        end = time.time()
        # print("time :",end-st)
        time_array.append(end - st)

        # Plot results
        def chunk_list(l, n):
            for i in range(0, len(l), n):
                yield l[i:i + n]

        size = int(episodes / 50)
        chunks = list(chunk_list(rewards, size))
        averages = [sum(chunk) / len(chunk) for chunk in chunks]
        size_array.append(size)
        chunks_array.append(chunks)
        averages_array.append(averages)
    return Q_array, epsilons, averages_array, reward_array, size_array, time_array


def perform_q_learning_varying_gamma(env, environment):
    ### Q-LEARNING #####
    print('Q LEARNING WITH FROZEN LAKE')
    st = time.time()
    reward_array = []
    iter_array = []
    size_array = []
    chunks_array = []
    averages_array = []
    time_array = []
    Q_array = []
    epsilons = [0.05, 0.15, 0.25, 0.5, 0.75, 0.90]
    gammas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for gamma in gammas:
        Q = np.zeros((env.observation_space.n, env.action_space.n))
        rewards = []
        iters = []
        optimal = [0] * env.observation_space.n
        alpha = 0.85
        epsilon = 0.75
        #gamma = 0.95
        episodes = 30000
        env = gym.make(environment)
        env = env.unwrapped
        desc = env.unwrapped.desc
        for episode in range(episodes):
            state = env.reset()
            done = False
            t_reward = 0
            max_steps = 1000000
            for i in range(max_steps):
                if done:
                    break
                current = state
                if np.random.rand() < (epsilon):
                    action = np.argmax(Q[current, :])
                else:
                    action = env.action_space.sample()

                state, reward, done, info = env.step(action)
                t_reward += reward
                Q[current, action] += alpha * (reward + gamma * np.max(Q[state, :]) - Q[current, action])
            epsilon = (1 - 2.71 ** (-episode / 1000))
            rewards.append(t_reward)
            iters.append(i)

        for k in range(env.observation_space.n):
            optimal[k] = np.argmax(Q[k, :])

        reward_array.append(rewards)
        iter_array.append(iters)
        Q_array.append(Q)

        env.close()
        end = time.time()
        # print("time :",end-st)
        time_array.append(end - st)

        # Plot results
        def chunk_list(l, n):
            for i in range(0, len(l), n):
                yield l[i:i + n]

        size = int(episodes / 50)
        chunks = list(chunk_list(rewards, size))
        averages = [sum(chunk) / len(chunk) for chunk in chunks]
        size_array.append(size)
        chunks_array.append(chunks)
        averages_array.append(averages)
    return Q_array, gammas, averages_array, reward_array, size_array, time_array

def perform_value_iteration(desc, env, max_tries):
    ### VALUE ITERATION ###
    print('VALUE ITERATION WITH FROZEN LAKE')
    time_array = [0] * max_tries
    gamma_arr = [0] * max_tries
    iters = [0] * max_tries
    list_scores = [0] * max_tries
    best_vals = [0] * max_tries
    for i in range(0, max_tries):
        st = time.time()
        best_value, k = value_iteration(env, gamma=(i + 0.5) / max_tries)
        policy = extract_policy(env, best_value, gamma=(i + 0.5) / max_tries)
        policy_score = evaluate_policy(env, policy, gamma=(i + 0.5) / max_tries, n=1000)
        gamma = (i + 0.5) / max_tries
        plot = plot_policy_map(
            'Frozen Lake Policy Map Iteration ' + str(i) + ' (Value Iteration) ' + 'Gamma: ' + str(gamma),
            policy.reshape(4, 4), desc, colors_lake(), directions_lake())
        end = time.time()
        gamma_arr[i] = (i + 0.5) / max_tries
        iters[i] = k
        best_vals[i] = best_value
        list_scores[i] = np.mean(policy_score)
        time_array[i] = end - st
    return best_vals, gamma_arr, iters, list_scores, time_array


def perform_policy_iteration(env, max_tries):
    time_array = [0] * max_tries
    gamma_arr = [0] * max_tries
    iters = [0] * max_tries
    list_scores = [0] * max_tries
    ### POLICY ITERATION ####
    print('POLICY ITERATION WITH FROZEN LAKE')
    for i in range(0, max_tries):
        st = time.time()
        best_policy, k = policy_iteration(env, gamma=(i + 0.5) / max_tries)
        scores = evaluate_policy(env, best_policy, gamma=(i + 0.5) / max_tries)
        end = time.time()
        gamma_arr[i] = (i + 0.5) / max_tries
        list_scores[i] = np.mean(scores)
        iters[i] = k
        time_array[i] = end - st
    return gamma_arr, iters, list_scores, time_array


def run_episode(env, policy, gamma, render=True):
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    while True:
        if render:
            env.render()
        obs, reward, done, _ = env.step(int(policy[obs]))
        total_reward += (gamma ** step_idx * reward)
        step_idx += 1
        if done:
            break
    return total_reward


def evaluate_policy(env, policy, gamma, n=100):
    scores = [run_episode(env, policy, gamma, False) for _ in range(n)]
    return np.mean(scores)


def extract_policy(env, v, gamma):
    policy = np.zeros(env.nS)
    for s in range(env.nS):
        q_sa = np.zeros(env.nA)
        for a in range(env.nA):
            q_sa[a] = sum([p * (r + gamma * v[s_]) for p, s_, r, _ in env.P[s][a]])
        policy[s] = np.argmax(q_sa)
    return policy


def compute_policy_v(env, policy, gamma):
    v = np.zeros(env.nS)
    eps = 1e-5
    while True:
        prev_v = np.copy(v)
        for s in range(env.nS):
            policy_a = policy[s]
            v[s] = sum([p * (r + gamma * prev_v[s_]) for p, s_, r, is_done in env.P[s][policy_a]])
        if (np.sum((np.fabs(prev_v - v))) <= eps):
            break
    return v


def policy_iteration(env, gamma):
    policy = np.random.choice(env.nA, size=(env.nS))
    max_iters = 200000
    desc = env.unwrapped.desc
    for i in range(max_iters):
        old_policy_v = compute_policy_v(env, policy, gamma)
        new_policy = extract_policy(env, old_policy_v, gamma)
        # if i % 2 == 0:
        #	plot = plot_policy_map('Frozen Lake Policy Map Iteration '+ str(i) + 'Gamma: ' + str(gamma),new_policy.reshape(4,4),desc,colors_lake(),directions_lake())
        #	a = 1
        if (np.all(policy == new_policy)):
            k = i + 1
            break
        policy = new_policy
    return policy, k


def value_iteration(env, gamma):
    v = np.zeros(env.nS)  # initialize value-function
    max_iters = 100000
    # todo changed
    eps = 1e-5 # 1e-20
    desc = env.unwrapped.desc
    for i in range(max_iters):
        prev_v = np.copy(v)
        for s in range(env.nS):
            q_sa = [sum([p * (r + gamma * prev_v[s_]) for p, s_, r, _ in env.P[s][a]]) for a in range(env.nA)]
            v[s] = max(q_sa)
        # if i % 50 == 0:
        #	plot = plot_policy_map('Frozen Lake Policy Map Iteration '+ str(i) + ' (Value Iteration) ' + 'Gamma: '+ str(gamma),v.reshape(4,4),desc,colors_lake(),directions_lake())
        if (np.sum(np.fabs(prev_v - v)) <= eps):
            k = i + 1
            break
    return v, k


def perform_q_learning_mdp(states):
    print('Q LEARNING WITH FOREST MANAGEMENT')
    P, R = mdptoolbox.example.forest(S=states, p=0.01)
    value_f = []
    policy = []
    iters = []
    time_array = []
    Q_table = []
    rew_array = []
    epsilons = [0.05, 0.15, 0.25, 0.5, 0.75, 0.95]
    gammas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for epsilon in epsilons:
        st = time.time()
        pi = mdp.QLearning(P, R, 0.95)
        end = time.time()
        pi.run(epsilon)
        rew_array.append(pi.reward_array)
        value_f.append(np.mean(pi.V))
        policy.append(pi.policy)
        time_array.append(end - st)
        Q_table.append(pi.Q)
    return Q_table, rew_array, epsilons, time_array, time_array


def perform_q_learning_mdp_with_varying_gamma(states):
    print('Q LEARNING WITH FOREST MANAGEMENT')
    P, R = mdptoolbox.example.forest(S=states, p=0.01)
    value_f = []
    policy = []
    iters = []
    time_array = []
    Q_table = []
    rew_array = []
    epsilons = [0.05, 0.15, 0.25, 0.5, 0.75, 0.95]
    epsilon = 0.75
    gammas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for gamma in gammas:
        st = time.time()
        pi = mdp.QLearning(P, R, gamma)
        end = time.time()
        pi.run(epsilon)
        rew_array.append(pi.reward_array)
        value_f.append(np.mean(pi.V))
        policy.append(pi.policy)
        time_array.append(end - st)
        Q_table.append(pi.Q)
    return Q_table, rew_array, gammas, time_array, time_array


def perform_value_iteration_mdp(states):
    print('VALUE ITERATION WITH FOREST MANAGEMENT')
    P, R = mdptoolbox.example.forest(S=states)
    value_f = [0] * 10
    policy = [0] * 10
    iters = [0] * 10
    time_array = [0] * 10
    gamma_arr = [0] * 10
    for i in range(0, 10):
        pi = mdptoolbox.mdp.ValueIteration(P, R, (i + 0.5) / 10)
        pi.run()
        gamma_arr[i] = (i + 0.5) / 10
        value_f[i] = np.mean(pi.V)
        policy[i] = pi.policy
        iters[i] = pi.iter
        time_array[i] = pi.time
    return gamma_arr, iters, time_array, value_f


def perform_policy_iteration_mdp(states):
    print('POLICY ITERATION WITH FOREST MANAGEMENT')
    P, R = mdptoolbox.example.forest(S=states)
    value_f = [0] * 10
    policy = [0] * 10
    iters = [0] * 10
    time_array = [0] * 10
    gamma_arr = [0] * 10
    for i in range(0, 10):
        pi = mdptoolbox.mdp.PolicyIteration(P, R, (i + 0.5) / 10)
        pi.run()
        gamma_arr[i] = (i + 0.5) / 10
        value_f[i] = np.mean(pi.V)
        policy[i] = pi.policy
        iters[i] = pi.iter
        time_array[i] = pi.time
    return gamma_arr, iters, time_array, value_f


def colors_lake():
    return {
        b'S': 'green',
        b'F': 'skyblue',
        b'H': 'black',
        b'G': 'gold',
    }


def directions_lake():
    return {
        3: '⬆',
        2: '➡',
        1: '⬇',
        0: '⬅'
    }


def actions_taxi():
    return {
        0: '⬇',
        1: '⬆',
        2: '➡',
        3: '⬅',
        4: 'P',
        5: 'D'
    }


def colors_taxi():
    return {
        b'+': 'red',
        b'-': 'green',
        b'R': 'yellow',
        b'G': 'blue',
        b'Y': 'gold'
    }


print('STARTING EXPERIMENTS')
output_path = "output"
perform_frozen_lake_experiments(output_path)
perform_forest_experiments(output_path)
print('END OF EXPERIMENTS')

