from agent.evaluation import plot_learning_curve, performance_matrix, plot_AFS_heatmap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mtick
import pandas as pd
import os
import math


#  Create plots
# --------------
# Set font size for all plots
plt.rcParams['font.size'] = '20'

# Format plot labels
# Code from https://stackoverflow.com/questions/59969492/how-to-print-10k-20k-1m-in-the-xlabel-of-matplotlib-plot
def format_func(value, tick_number=None):
    num_thousands = 0 if abs(value) < 1000 else math.floor (math.log10(abs(value))/3)
    value = round(value / 1000**num_thousands, 2)
    return f'{value:g}'+' KMGTPEZY'[num_thousands]

# Single task results (baseline)
print("Single task plots")
for s in ["perpendicular", "diagonal-25", "diagonal-50", "parallel"]:
    plot_learning_curve(plot_path = f"./plots/{s}", 
                        data_paths = {"random" : "./results/random_baseline/data", 
                                      "sac" : f"./results/sac/{s}/data", 
                                      "ppo" : f"./results/ppo/{s}/data", 
                                      "drama" : f"./results/drama/{s}/data"}, 
                        agent_list = ["sac", "ppo", "drama"],
                        plt_kwargs = {"sac"    : {"linestyle" : "-", "color" : "#008800", "linewidth" : 2}, 
                                      "ppo"    : {"linestyle" : "--", "color" : "#ee0000", "linewidth" : 2}, 
                                      "drama"  : {"linestyle" : ":", "color" : "blue", "linewidth" : 2.5},
                                     },
                        max_steps=2_000_000, 
                        task_interval=500_000, 
                        metrics=["reward", "success", "crashed", "truncated"],
                        scenarios = [s], subplot_height=3, save_format="eps", plt_std=False)
plt.close('all')

# PPO parallel parking
print("PPO parallel parking plots")
plot_learning_curve(plot_path = "./plots/parallel_1M", 
                    data_paths = {"random" : "./results/random_baseline/data", 
                                  "ppo" : "./results/ppo/parallel-1M/data", 
                                 }, 
                    agent_list = ["ppo"],
                    plt_kwargs = {"ppo"    : {"linestyle" : "-"}, "color" : "black"},
                    max_steps=2_000_000, 
                    task_interval=2_000_000, 
                    metrics=["reward", "success", "crashed", "truncated"],
                    scenarios = ["parallel"], subplot_height=3, subplot_width=15, save_format="eps", plt_std=False)
plt.close('all')

plot_learning_curve(plot_path = "./plots/parallel-adj_1M", 
                    data_paths = {"random" : "./results/random_baseline/data", 
                                  "ppo" : "./results/ppo/parallel-adj-1M/data", 
                                 }, 
                    agent_list = ["ppo"],
                    plt_kwargs = {"ppo"    : {"linestyle" : "-", "color" : "black"},},
                    max_steps=2_000_000, 
                    task_interval=2_000_000, 
                    metrics=["reward", "success", "crashed", "truncated"],
                    scenarios = ["parallel-adj"], subplot_height=3, subplot_width=15, save_format="eps", plt_std=False)
plt.close('all')

plot_learning_curve(plot_path = "./plots/parallel-adj-tuned_1M", 
                    data_paths = {"random" : "./results/random_baseline/data", 
                                  "ppo-tuned" : "./results/ppo/parallel-adj-tuned-1M/data", 
                                 }, 
                    agent_list = ["ppo-tuned"],
                    plt_kwargs = {"ppo-tuned" : {"linestyle" : "-", "color" : "black"},},
                    max_steps=2_000_000, 
                    task_interval=2_000_000, 
                    metrics=["reward", "success", "crashed", "truncated"],
                    scenarios = ["parallel-adj"], subplot_height=3, subplot_width=15, save_format="eps", plt_std=False)
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
                    plt_kwargs = {"sac"    : {"linestyle" : "-", "color" : "#008800", "linewidth" : 2}, 
                                  "ppo"    : {"linestyle" : "--", "color" : "#ee0000", "linewidth" : 2}, 
                                  "drama"  : {"linestyle" : ":", "color" : "blue", "linewidth" : 2.5},
                                 },
                    max_steps=2_000_000, 
                    task_interval=2_000_000, 
                    metrics=["reward", "success", "crashed", "truncated"],
                    scenarios = ["perpendicular", "diagonal-25", "diagonal-50", "parallel"], subplot_height=3, subplot_width=15, save_format="eps", plt_std=False)
plt.close('all')

# Sequential scenarios
print("Sequential task plots")
plot_learning_curve(plot_path = "./plots/sequential-inc", 
                    data_paths = {"random" : "./results/random_baseline/data", 
                                  "sac" : "./results/sac/sequential-inc/data", 
                                  "ppo" : "./results/ppo/sequential-inc/data", 
                                  "drama" : "./results/drama/sequential-inc/data"}, 
                    agent_list = ["sac", "ppo", "drama"],
                    plt_kwargs = {"sac"    : {"linestyle" : "-", "color" : "#008800", "linewidth" : 2}, 
                                  "ppo"    : {"linestyle" : "--", "color" : "#ee0000", "linewidth" : 2}, 
                                  "drama"  : {"linestyle" : ":", "color" : "blue", "linewidth" : 2.5},
                                 },
                    max_steps=2_000_000, 
                    task_interval=500_000, 
                    metrics=["reward", "success", "crashed", "truncated"],
                    scenarios = ["perpendicular", "diagonal-25", "diagonal-50", "parallel"], subplot_height=3, subplot_width=15,
                    plot_envs="sequential-inc",
                    env_np_path = "./results/env_images", save_format="eps", plt_std=False)
plt.close('all')

plot_learning_curve(plot_path = "./plots/sequential-dec", 
                    data_paths = {"random" : "./results/random_baseline/data", 
                                  "sac" : "./results/sac/sequential-dec/data", 
                                  "ppo" : "./results/ppo/sequential-dec/data", 
                                  "drama" : "./results/drama/sequential-dec/data"}, 
                    agent_list = ["sac", "ppo", "drama"],
                    plt_kwargs = {"sac"    : {"linestyle" : "-", "color" : "#008800", "linewidth" : 2}, 
                                  "ppo"    : {"linestyle" : "--", "color" : "#ee0000", "linewidth" : 2}, 
                                  "drama"  : {"linestyle" : ":", "color" : "blue", "linewidth" : 2.5},
                                 },
                    max_steps=2_000_000, 
                    task_interval=500_000, 
                    metrics=["reward", "success", "crashed", "truncated"],
                    scenarios = ["perpendicular", "diagonal-25", "diagonal-50", "parallel"],
                    plot_envs="sequential-dec", subplot_height=3, subplot_width=15,
                    env_np_path = "./results/env_images", save_format="eps", plt_std=False)
plt.close('all')

# Plot PPO+EWC
plot_learning_curve(plot_path = "./plots/ppo+ewc", 
                    data_paths = {"random" : "./results/random_baseline/data", 
                                  "ppo" : "./results/ppo/sequential-inc/data", 
                                  "ppo-ewc" : "./results/ppo-ewc/sequential-inc_lambda-1_discount-0.5/data"},
                    plt_kwargs = {"ppo"    : {"linestyle" : "-", "color" : "black"}, 
                                  "ppo-ewc"    : {"linestyle" : "--", "color" : "black"}, 
                                 },
                    agent_list = ["ppo", "ppo-ewc"],
                    max_steps=2_000_000, 
                    task_interval=500_000, 
                    metrics=["reward", "success", "crashed", "truncated"],
                    scenarios = ["perpendicular", "diagonal-25", "diagonal-50", "parallel"],
                    plot_envs="sequential-inc", subplot_height=3, subplot_width=15,
                    env_np_path = "./results/env_images", save_format="eps", plt_std=False)
plt.close('all')

##############

paths = {"ppo" : "./results/ppo/parallel-1M/data/parallel_success_mean.csv",
         "ppo+adj-rewards" : "./results/ppo/parallel-adj-1M/data/parallel-adj_success_mean.csv",
         "ppo+adj-rewards+tuned" : "./results/ppo/parallel-adj-tuned-1M/data/parallel-adj_success_mean.csv",
        }

plt_kwargs = {"ppo" : {"linestyle" : "-", "color" : "#008800", "linewidth" : 2}, 
              "ppo+adj-rewards" : {"linestyle" : "--", "color" : "#ee0000", "linewidth" : 2}, 
              "ppo+adj-rewards+tuned" : {"linestyle" : ":", "color" : "blue", "linewidth" : 2.5}, 
             }

fig, ax = plt.subplots(figsize=(15, 3))

for agent, path in paths.items():
    # Read data
    #path = paths[agent] + f"/{scenarios[i]}_{m}_mean.csv"
    df = pd.read_csv(path)
                            
    # Calculate mean and std
    x = df["idx"]
    y = df[df.columns[1:]].mean(axis="columns")
    std = df[df.columns[1:]].std(axis="columns")

    # Plot mean and std
    ax.plot(x, y, label=agent, **plt_kwargs[agent])

# Format y-axis
ymin = 0
ymax = 1
ybuff = 0.1*(ymax-ymin)
ax.set_ylim((ymin-ybuff, ymax+ybuff))
ax.set_ylabel(f"Success rate")
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

# Format x-axis
xmin = 0
xmax = 1_000_000
ax.set_xlim((xmin, xmax))
ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
ax.set_xlabel("Environment steps")
                
# Add legend to top plot
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Create missing directories
plot_path = "./plots/ppo-parallel"
if not os.path.exists(plot_path):
    print(f"Creating output directory {plot_path}")
    os.makedirs(plot_path)
fig.tight_layout()
fig.savefig(f"{plot_path}/learning_curve_success.eps", format="eps")
fig.savefig(f"{plot_path}/learning_curve_success.png", format="png")


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
        plot_AFS_heatmap(path, scale_all=True, save_format="eps")
        plot_AFS_heatmap(path, scale_all=False, save_format="eps")
        