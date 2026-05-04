from agent.evaluation import plot_learning_curve, performance_matrix
import numpy as np
import matplotlib.pyplot as plt

"""
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
                        max_steps=2_000_000, 
                        task_interval=500_000, 
                        metrics=["reward", "success", "crashed", "truncated"],
                        scenarios = [s], subplot_height=5)
plt.close('all')

# Interleaved
print("Interleaved task plots")
plot_learning_curve(plot_path = "./plots/interleave", 
                    data_paths = {"random" : "./results/random_baseline/data", 
                                  "sac" : "./results/sac/interleave/data", 
                                  "ppo" : "./results/ppo/interleave/data", 
                                  "drama" : "./results/drama/interleave/data"}, 
                    max_steps=2_000_000, 
                    task_interval=500_000, 
                    metrics=["reward", "success", "crashed", "truncated"],
                    scenarios = ["perpendicular", "diagonal-25", "diagonal-50", "parallel"])
plt.close('all')

# Sequential scenarios
print("Sequential task plots")
plot_learning_curve(plot_path = "./plots/sequential-inc", 
                    data_paths = {"random" : "./results/random_baseline/data", 
                                  "sac" : "./results/sac/sequential-inc/data", 
                                  "ppo" : "./results/ppo/sequential-inc/data", 
                                  "drama" : "./results/drama/sequential-inc/data"}, 
                    max_steps=2_000_000, 
                    task_interval=500_000, 
                    metrics=["reward", "success", "crashed", "truncated"],
                    scenarios = ["perpendicular", "diagonal-25", "diagonal-50", "parallel"])

plot_learning_curve(plot_path = "./plots/sequential-dec", 
                    data_paths = {"random" : "./results/random_baseline/data", 
                                  "sac" : "./results/sac/sequential-dec/data", 
                                  "ppo" : "./results/ppo/sequential-dec/data", 
                                  "drama" : "./results/drama/sequential-dec/data"}, 
                    max_steps=2_000_000, 
                    task_interval=500_000, 
                    metrics=["reward", "success", "crashed", "truncated"],
                    scenarios = ["perpendicular", "diagonal-25", "diagonal-50", "parallel"])
plt.close('all')


#  Calculate BWT and FWT
# -----------------------

for metric in ["reward", "success"]:
    file = open(f"./plots/transfer-learning-metrics_{metric}.txt", "w")
    for alg in ["ppo", "sac", "drama"]:
        for scenario in ["sequential-dec", "sequential-inc"]:
    
            matrix = np.loadtxt(f"./results/{alg}/{scenario}/data/average_performance_matrix-{metric}.csv", delimiter=",")
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
    
            print(alg, scenario, "BWT:", bwt, "; FWT:", fwt)
    
            file.write(f"Algorithm: {alg}; Scenario: {scenario}\n") 
            file.write(str(matrix) + "\n")
            file.write(f"BWT: {bwt}\n") 
            file.write(f"FWT: {fwt}\n\n") 
            
    file.close()
"""
        





        