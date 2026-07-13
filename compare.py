from agent.evaluation import plot_learning_curve, performance_matrix, plot_AFS_heatmap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


#  Create plots
# --------------

# Single task results (baseline)
print("Single task plots")
for s in ["perpendicular", "diagonal-25", "diagonal-50", "parallel"]:
    plot_learning_curve(plot_path = f"./plots/{s}", 
                        data_paths = {"random" : "./results/random_baseline/data", 
                                      "sac" : f"./results/sac/{s}/data", 
                                      "ppo" : f"./results/ppo/{s}/data", 
                                      "drama" : f"./results/drama/{s}/data"}, 
                        agent_list = ["sac", "ppo", "drama"],
                        plt_kwargs = {"sac"    : {"color" : mcolors.TABLEAU_COLORS["tab:blue"]}, 
                                      "ppo"    : {"color" : mcolors.TABLEAU_COLORS["tab:orange"]}, 
                                      "drama"  : {"color" : mcolors.TABLEAU_COLORS["tab:green"]},
                                     },
                        max_steps=2_000_000, 
                        task_interval=500_000, 
                        metrics=["reward", "success", "crashed", "truncated"],
                        scenarios = [s], subplot_height=3, save_format="png")
plt.close('all')

# PPO parallel parking
print("PPO parallel parking plots")
plot_learning_curve(plot_path = "./plots/parallel_1M", 
                    data_paths = {"random" : "./results/random_baseline/data", 
                                  "ppo" : "./results/ppo/parallel-1M/data", 
                                 }, 
                    agent_list = ["ppo"],
                    plt_kwargs = {"ppo"    : {"color" : mcolors.TABLEAU_COLORS["tab:orange"]}, },
                    max_steps=2_000_000, 
                    task_interval=2_000_000, 
                    metrics=["reward", "success", "crashed", "truncated"],
                    scenarios = ["parallel"], save_format="png")
plt.close('all')

plot_learning_curve(plot_path = "./plots/parallel-adj_1M", 
                    data_paths = {"random" : "./results/random_baseline/data", 
                                  "ppo" : "./results/ppo/parallel-adj-1M/data", 
                                 }, 
                    agent_list = ["ppo"],
                    plt_kwargs = {"ppo"    : {"color" : mcolors.TABLEAU_COLORS["tab:orange"]},},
                    max_steps=2_000_000, 
                    task_interval=2_000_000, 
                    metrics=["reward", "success", "crashed", "truncated"],
                    scenarios = ["parallel-adj"], save_format="png")
plt.close('all')

plot_learning_curve(plot_path = "./plots/parallel-adj-tuned_1M", 
                    data_paths = {"random" : "./results/random_baseline/data", 
                                  "ppo-tuned" : "./results/ppo/parallel-adj-tuned-1M/data", 
                                 }, 
                    agent_list = ["ppo-tuned"],
                    plt_kwargs = {"ppo-tuned" : {"color" : mcolors.TABLEAU_COLORS["tab:orange"]},},
                    max_steps=2_000_000, 
                    task_interval=2_000_000, 
                    metrics=["reward", "success", "crashed", "truncated"],
                    scenarios = ["parallel-adj"], save_format="png")
plt.close('all')

# Interleaved
print("Interleaved task plots")
plot_learning_curve(plot_path = "./plots/interleave_2M", 
                    data_paths = {"random" : "./results/random_baseline/data", 
                                  "sac" : "./results/sac/interleave_2M/data", 
                                  "ppo" : "./results/ppo/interleave_2M/data", 
                                  "drama" : "./results/drama/interleave_2M/data"
                                 }, 
                    agent_list = ["sac", "ppo", "drama"],
                    plt_kwargs = {"sac"    : {"color" : mcolors.TABLEAU_COLORS["tab:blue"]}, 
                                  "ppo"    : {"color" : mcolors.TABLEAU_COLORS["tab:orange"]}, 
                                  "drama"  : {"color" : mcolors.TABLEAU_COLORS["tab:green"]},
                                 },
                    max_steps=2_000_000, 
                    task_interval=2_000_000, 
                    metrics=["reward", "success", "crashed", "truncated"],
                    scenarios = ["perpendicular", "diagonal-25", "diagonal-50", "parallel"], save_format="png")
plt.close('all')

# Sequential scenarios
print("Sequential task plots")
plot_learning_curve(plot_path = "./plots/sequential-inc", 
                    data_paths = {"random" : "./results/random_baseline/data", 
                                  "sac" : "./results/sac/sequential-inc/data", 
                                  "ppo" : "./results/ppo/sequential-inc/data", 
                                  "drama" : "./results/drama/sequential-inc/data"}, 
                    agent_list = ["sac", "ppo", "drama"],
                    plt_kwargs = {"sac"    : {"color" : mcolors.TABLEAU_COLORS["tab:blue"]}, 
                                  "ppo"    : {"color" : mcolors.TABLEAU_COLORS["tab:orange"]}, 
                                  "drama"  : {"color" : mcolors.TABLEAU_COLORS["tab:green"]},
                                 },
                    max_steps=2_000_000, 
                    task_interval=500_000, 
                    metrics=["reward", "success", "crashed", "truncated"],
                    scenarios = ["perpendicular", "diagonal-25", "diagonal-50", "parallel"],
                    plot_envs="sequential-inc",
                    env_np_path = "./results/env_images", save_format="png")
plt.close('all')

plot_learning_curve(plot_path = "./plots/sequential-dec", 
                    data_paths = {"random" : "./results/random_baseline/data", 
                                  "sac" : "./results/sac/sequential-dec/data", 
                                  "ppo" : "./results/ppo/sequential-dec/data", 
                                  "drama" : "./results/drama/sequential-dec/data"}, 
                    agent_list = ["sac", "ppo", "drama"],
                    plt_kwargs = {"sac"    : {"color" : mcolors.TABLEAU_COLORS["tab:blue"]}, 
                                  "ppo"    : {"color" : mcolors.TABLEAU_COLORS["tab:orange"]}, 
                                  "drama"  : {"color" : mcolors.TABLEAU_COLORS["tab:green"]},
                                 },
                    max_steps=2_000_000, 
                    task_interval=500_000, 
                    metrics=["reward", "success", "crashed", "truncated"],
                    scenarios = ["perpendicular", "diagonal-25", "diagonal-50", "parallel"],
                    plot_envs="sequential-dec",
                    env_np_path = "./results/env_images", save_format="png")
plt.close('all')

# Plot PPO+EWC
plot_learning_curve(plot_path = "./plots/ppo+ewc", 
                    data_paths = {"random" : "./results/random_baseline/data", 
                                  "ppo" : "./results/ppo/sequential-inc/data", 
                                  "ppo-ewc" : "./results/ppo-ewc/sequential-inc_lambda-1_discount-0.5/data"},
                    plt_kwargs = {"ideal" : {"color" : mcolors.TABLEAU_COLORS["tab:green"]},
                                  "random" : {"color" : mcolors.TABLEAU_COLORS["tab:red"]}, 
                                  "ppo-ewc" : {"color" : mcolors.TABLEAU_COLORS["tab:blue"]}, 
                                  "ppo" : {"color" : mcolors.TABLEAU_COLORS["tab:orange"]}, 
                                 },
                    agent_list = ["ppo", "ppo-ewc"],
                    max_steps=2_000_000, 
                    task_interval=500_000, 
                    metrics=["reward", "success", "crashed", "truncated"],
                    scenarios = ["perpendicular", "diagonal-25", "diagonal-50", "parallel"],
                    plot_envs="sequential-inc",
                    env_np_path = "./results/env_images", save_format="png")
plt.close('all')



#  Calculate BWT and FWT
# -----------------------

def calculate_bwt_fwt(out_file, in_path, header):
    matrix = np.loadtxt(in_path, delimiter=",")
    print()
    
    # BWT=1/(T-1)*\sum^{T-1}_{i=1}R_{T,i}-R_{i,i}
    T = len(matrix)
    s = 0
    for i in range(0,T-1):
        print(i, matrix[i][T] - matrix[i][i+1])
        s += matrix[i][T] - matrix[i][i+1]
    bwt = s / (T-1)
    
    # FWT=1/(T-1)*\sum_{i=2}^TR_{i-1,i}-\overline{b}_i
    s = 0
    for i in range(1,T):
        print(i, matrix[i][i] - matrix[i][0])
        s += matrix[i][i] - matrix[i][0]
    fwt = s / (T-1)
    
    print(header, "BWT:", bwt, "; FWT:", fwt)
    
    file.write(f"{header}\n") 
    file.write(str(matrix) + "\n")
    file.write(f"BWT: {bwt}\n") 
    file.write(f"FWT: {fwt}\n\n") 


# Matrix for forward and backward transfer metrics
for alg in ["ppo", "sac", "drama"]:
    for scenario in ["sequential-inc", "sequential-dec"]:
        for metric in ["reward", "success"]:
            performance_matrix(f"./results/{alg}/{scenario}" + "/data", scenario, 500_000, performance_metric=metric)


for metric in ["reward", "success"]:
    file = open(f"./plots/transfer-learning-metrics_{metric}.txt", "w")
    for alg in ["ppo", "sac", "drama"]:
        for scenario in ["sequential-dec", "sequential-inc"]:
            header = f"Algorithm: {alg}; Scenario: {scenario}"
            in_path = f"./results/{alg}/{scenario}/data/average_performance_matrix-{metric}.csv"
            calculate_bwt_fwt(file, in_path, header)
    
    file.close()

# PPO+EWC BWT and FWT
for metric in ["reward", "success"]:
    file = open(f"./plots/transfer-learning-metrics_{metric}.txt", "a")

    configs = [{"folder" : "sequential-inc", "lambda" : 0.5, "discount" : 0.9},
               {"folder" : "sequential-inc_lambda-1", "lambda" : 1, "discount" : 0.9},
               {"folder" : "sequential-inc_lambda-10", "lambda" : 10, "discount" : 0.9},
               {"folder" : "sequential-inc_lambda-1_discount-0.1", "lambda" : 1, "discount" : 0.1},
               {"folder" : "sequential-inc_lambda-1_discount-0.5", "lambda" : 1, "discount" : 0.5},]
    
    for c in configs:
            header = f"Algorithm: PPO+EWC; Scenario: sequential-inc, lambda: {c['lambda']}, discount: {c['discount']}"
            in_path = f"./results/ppo-ewc/{c['folder']}/data/average_performance_matrix-{metric}.csv"
            calculate_bwt_fwt(file, in_path, header)
    
    file.close()

for alg in ["ppo", "sac"]:
    for scenario in ["perpendicular", "diagonal-25", "diagonal-50", "parallel", "sequential-dec", "sequential-inc"]:
        print(f"\n{alg} {scenario}")
        path = f"./results/{alg}/{scenario}"
        plot_AFS_heatmap(path, scale_all=True, save_format="png")
        plot_AFS_heatmap(path, scale_all=False, save_format="png")
        