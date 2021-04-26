from matplotlib import pyplot as plt


def plot_qlearning_experiment_results_forest_forest(output_path, Q_table, algorithm, experiment_name, rew_array, epsilons,time_array ):
    for i in range(len(rew_array)):
        plt.plot(range(0, 10000), rew_array[i], label=f'epsilon={epsilons[i]}')

    title = f'{experiment_name} ({algorithm}) - Reward vs Iterations'
    plt.legend()
    plt.xlabel('Iterations')
    plt.grid()
    plt.title(title)
    plt.ylabel('Average Reward')
    # plt.show()
    plt.savefig(fr"{output_path}\{title}.png")
    plt.close()

    title = f'{experiment_name} ({algorithm}) - Epsilon vs Execution Time'
    plt.plot(epsilons, time_array)
    plt.xlabel('Epsilon')
    plt.grid()
    plt.title(title)
    plt.ylabel('Execution Time (s)')
    # plt.show()
    plt.savefig(fr"{output_path}\{title}.png")
    plt.close()

    for i in range(len(rew_array)):
        plt.subplot(1, 6, i+1)
        plt.imshow(Q_table[i][:20, :])
        plt.title(f'Eps={epsilons[i]}')

    plt.colorbar()
    # plt.show()
    plt.savefig(fr"{output_path}\{experiment_name} - {algorithm} - Epsilon Comparison.png")
    plt.close()

def plot_qlearning_experiment_results_forest_forest_with_gammas(output_path, Q_table, algorithm, experiment_name, rew_array, gammas,time_array ):
    for i in range(len(rew_array)):
        plt.plot(range(0, 10000), rew_array[i], label=f'gamma={gammas[i]}')

    title = f'{experiment_name} ({algorithm}) - Reward vs Iterations'
    plt.legend()
    plt.xlabel('Iterations')
    plt.grid()
    plt.title(title)
    plt.ylabel('Average Reward')
    # plt.show()
    plt.savefig(fr"{output_path}\{title}.png")
    plt.close()

    title = f'{experiment_name} ({algorithm}) - Gamma vs Execution Time'
    plt.plot(gammas, time_array)
    plt.xlabel('Gamma')
    plt.grid()
    plt.title(title)
    plt.ylabel('Execution Time (s)')
    # plt.show()
    plt.savefig(fr"{output_path}\{title}.png")
    plt.close()


def plot_iteration_experiment_results_forest(output_path, algorithm, experiment_name, gamma_arr, iters, time_array, value_f):
    title = f'{experiment_name} ({algorithm}) - Execution Time Vs Gamma'
    plt.plot(gamma_arr, time_array)
    plt.xlabel('Gamma')
    plt.title(title)
    plt.ylabel('Execution Time (s)')
    plt.grid()
    # plt.show()
    plt.savefig(fr"{output_path}\0{title}.png")
    plt.close()

    title = f'{experiment_name} ({algorithm}) - Reward vs Gamma'
    plt.plot(gamma_arr, value_f)
    plt.xlabel('Gamma')
    plt.ylabel('Average Rewards')
    plt.title(title)
    plt.grid()
    # plt.show()
    plt.savefig(fr"{output_path}\1{title}.png")
    plt.close()

    title = f'{experiment_name} ({algorithm}) - Convergence Iterations vs Gamma'
    plt.plot(gamma_arr, iters)
    plt.xlabel('Gamma')
    plt.ylabel('Iterations to Converge')
    plt.title(title)
    plt.grid()
    # plt.show()
    plt.savefig(fr"{output_path}\2{title}.png")
    plt.close()


def plot_q_learning_experiment_results(output_path, Q_array, algorithm, averages_array, epsilons, experiment_name, reward_array,
                                       size_array, time_array):

    for i in range(len(reward_array)):
        plt.plot(range(0, len(reward_array[i]), size_array[i]), averages_array[i], label=f'epsilon={epsilons[i]}')

    title = f'{experiment_name} ({algorithm}) - Reward vs Iterations'
    plt.legend()
    plt.xlabel('Iterations')
    plt.grid()
    plt.title(title)
    plt.ylabel('Average Reward')
    # plt.show()
    plt.savefig(fr"{output_path}\{title}.png")
    plt.close()

    title = f'{experiment_name} ({algorithm}) - Epsilon vs Execution Time'
    plt.plot(epsilons, time_array)
    plt.xlabel('Epsilon')
    plt.grid()
    plt.title(title)
    plt.ylabel('Execution Time (s)')
    # plt.show()
    plt.savefig(fr"{output_path}\{title}.png")
    plt.close()

    for i in range(len(reward_array)):
        plt.subplot(1, 6, i + 1)
        plt.imshow(Q_array[i])
        plt.title(f'Eps={epsilons[i]}')

    plt.tight_layout()
    plt.colorbar()
    plt.savefig(fr"{output_path}\{title}_epsilon_comparison.png")
    plt.close()


def plot_q_learning_experiment_results_with_gammas(output_path, Q_array, algorithm, averages_array, gammas, experiment_name, reward_array,
                                                    size_array, time_array):

    for i in range(len(reward_array)):
        plt.plot(range(0, len(reward_array[i]), size_array[i]), averages_array[i], label=f'gamma={gammas[i]}')

    title = f'{experiment_name} ({algorithm}) - Reward vs Iterations'
    plt.legend()
    plt.xlabel('Iterations')
    plt.grid()
    plt.title(title)
    plt.ylabel('Average Reward')
    # plt.show()
    plt.savefig(fr"{output_path}\{title}.png")
    plt.close()

    title = f'{experiment_name} ({algorithm}) - Gamma vs Execution Time'
    plt.plot(gammas, time_array)
    plt.xlabel('Gamma')
    plt.grid()
    plt.title(title)
    plt.ylabel('Execution Time (s)')
    # plt.show()
    plt.savefig(fr"{output_path}\{title}.png")
    plt.close()

    try:
        for i in range(len(reward_array)):
            plt.subplot(1, 6, i + 1)
            plt.imshow(Q_array[i])
            plt.title(f'g={gammas[i]}')

        plt.tight_layout()
        plt.colorbar()
        plt.savefig(fr"{output_path}\{title}_gamma_comparison.png")
        plt.close()
    except:
        pass


def plot_iteration_experiment_results(output_path, algorithm, experiment_name, gamma_arr, iters, list_scores, time_array, best_vals):
    try:
        title = f'{experiment_name} ({algorithm}) - Execution Time Vs Gamma'
        plt.plot(gamma_arr, time_array)
        plt.xlabel('Gamma')
        plt.title(title)
        plt.ylabel('Execution Time (s)')
        plt.grid()
        #plt.show()
        plt.savefig(fr"{output_path}\0{title}.png")
        plt.close()
    except:
        pass

    try:
        title = f'{experiment_name} ({algorithm}) - Reward vs Gamma'
        plt.plot(gamma_arr, list_scores)
        plt.xlabel('Gamma')
        plt.ylabel('Average Rewards')
        plt.title(title)
        plt.grid()
        # plt.show()
        plt.savefig(fr"{output_path}\1{title}.png")
        plt.close()
    except:
        pass

    try:
        title = f'{experiment_name} ({algorithm}) - Convergence Iterations vs Gamma'
        plt.plot(gamma_arr, iters)
        plt.xlabel('Gamma')
        plt.ylabel('Iterations to Converge')
        plt.title(title)
        plt.grid()
        # plt.show()
        plt.savefig(fr"{output_path}\2{title}.png")
        plt.close()
    except:
        pass

    try:
        title = f'{experiment_name} - {algorithm} - Best Value vs Gamma'
        plt.plot(gamma_arr, best_vals)
        plt.xlabel('Gamma')
        plt.ylabel('Optimal Value')
        plt.title(title)
        plt.grid()
        # plt.show()
        plt.savefig(fr"{output_path}\3{title}.png")
        plt.close()
    except:
        pass


def plot_policy_map(title, policy, map_desc, color_map, direction_map):
    fig = plt.figure()
    ax = fig.add_subplot(111, xlim=(0, policy.shape[1]), ylim=(0, policy.shape[0]))
    font_size = 'x-large'
    if policy.shape[1] > 16:
        font_size = 'small'
    plt.title(title)
    for i in range(policy.shape[0]):
        for j in range(policy.shape[1]):
            y = policy.shape[0] - i - 1
            x = j
            p = plt.Rectangle([x, y], 1, 1)
            p.set_facecolor(color_map[map_desc[i, j]])
            ax.add_patch(p)

            text = ax.text(x + 0.5, y + 0.5, direction_map[policy[i, j]], weight='bold', size=font_size,
                           horizontalalignment='center', verticalalignment='center', color='w')

    plt.axis('off')
    plt.xlim((0, policy.shape[1]))
    plt.ylim((0, policy.shape[0]))
    plt.tight_layout()
    #plt.savefig(title + '.png')
    plt.close()

    return (plt)