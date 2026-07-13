# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 15:31:07 2025

Functions and classes for evaluating agents. This includes:
    - Logging recorded metrics
    - Creating plots

@author: nollek
"""

import datetime as dt
import json
import os
import pandas as pd
import shutil
import numpy as np
import math


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mtick
import seaborn as sns

# Set font size for all plots
plt.rcParams['font.size'] = '16'

# Logging class to write recorded metrics to files
# -----------------------------------------------------------------------------

class Logger:

    def __init__(self, path, values):
        """
        path : Path to store data
        metrics : Array of metrics to store
        """

        self.path = path
        self.tmp_path = path + "/tmp"
        self.data_path = path + "/data"

        self.values = values

        self.tmp_files = {}

        # Create missing folders
        if not os.path.exists(self.path):
            print(f"Creating output directory {self.path}")
            os.makedirs(self.path)
        if not os.path.exists(self.tmp_path):
            os.makedirs(self.tmp_path)
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

    def open(self, header):
        
        # Create file paths
        self.tmp_files["idx"] = f"{self.tmp_path}/idx.txt"
        header_dict = {}
        for v in self.values:
            self.tmp_files[v] = f"{self.tmp_path}/{v}.txt"
            header_dict[v] = header
            
        # Write headers
        self.append("idx", header_dict, mode="w")

    def append(self, idx, values, mode="a"):
        with open(self.tmp_files["idx"], mode) as f:
            f.write(f"{idx}\n")
        for k, v in values.items():
            with open(self.tmp_files[k], mode) as f:
                f.write(f"{v}\n")

    def close(self):
        # Merge into data file
        for v in self.values:
            if os.path.isfile(f"{self.data_path}/{v}.csv"):
                # Copy old version into tmp
                shutil.copyfile(f"{self.data_path}/{v}.csv", f"{self.tmp_path}/{v}_old.csv")
                # Concat tmp file with data file
                with open(f"{self.tmp_path}/{v}_old.csv") as file_old, open(f"{self.tmp_path}/{v}.txt") as file_new, open(f"{self.data_path}/{v}.csv","w") as file_out:
                    for x,y in zip(file_old, file_new):
                        file_out.write(x.strip()+","+y.strip()+'\n')
            else:
                # Create data file
                with open(f"{self.tmp_path}/idx.txt") as file_ixd, open(f"{self.tmp_path}/{v}.txt") as file_val1, open(f"{self.data_path}/{v}.csv","w") as file_out:
                    for x,y in zip(file_ixd, file_val1):
                        file_out.write(x.strip()+","+y.strip()+'\n')

        # Delete tmp files
        shutil.rmtree(self.tmp_path)
        

# Calculate aggregated metrics
# -----------------------------------------------------------------------------

def average_fisher_sensitivity(data_path):
    for fisher_dir in [x  for x in os.listdir(data_path) if x.startswith("fisher-information")]:
        
        step = int(fisher_dir.split('-')[-1])
        afs_step = [] # List of the average fisher sensitivity for the current step
        for filename in [x for x in os.listdir(data_path+'/'+fisher_dir) if "_AFS" not in x and "_MinMaxScaled" not in x and x.endswith(".json")]:
            print(filename)
            with open(data_path+'/'+fisher_dir+'/'+filename) as f:
                raw_data = json.load(f)

            total = 0
            afs = {}
            for name in raw_data.keys():
                total += np.sum(raw_data[name])
            print("Total Fisher information:", total)
            for name in raw_data.keys():
                if total > 0:
                    afs[name] = raw_data[name] / total
                else:
                    afs[name] = np.array(raw_data[name])
                
            # Save as JSON
            with open(data_path+'/'+fisher_dir+'/'+filename.split('.')[0]+'_AFS.json', "w") as outfile:
                json.dump({k: afs[k].tolist() for k, v in afs.items()}, outfile)
                
            afs_step.append(afs)
            
        # Calculate average over all runs
        total_afs = None
        n = 0
        for run_afs in afs_step:
            if total_afs is None:
                total_afs = run_afs
            else:
                for (total_n, total_v), (run_n, run_v) in zip(total_afs.items(), run_afs.items()):
                    assert (total_n == run_n)
                    total_afs[total_n] = np.array(total_v) + np.array(run_v)
            n += 1
        for name in total_afs.keys():
            total_afs[name] = total_afs[name] / n
            
        # Save as JSON
        with open(f"{data_path}/average-fisher-sensitivity_step-{step}.json", "w") as outfile:
            json.dump({k: total_afs[k].tolist() for k, v in total_afs.items()}, outfile)

def average_fisher_minmaxscale(data_path):
    for fisher_dir in [x  for x in os.listdir(data_path) if x.startswith("fisher-information")]:
        
        step = int(fisher_dir.split('-')[-1])
        scaled_data_step = [] # List of the average fisher sensitivity for the current step
        for filename in [x for x in os.listdir(data_path+'/'+fisher_dir) if "_AFS" not in x and "_MinMaxScaled" not in x and x.endswith(".json")]:
            print(filename)
            with open(data_path+'/'+fisher_dir+'/'+filename) as f:
                raw_data = json.load(f)

            # Get maximum values
            vmax = 0
            for name in raw_data.keys():
                vmax = max(vmax, np.max(raw_data[name]))
                
            # Normalise importances (min max scaling to [0, 1] with min=0)
            scaled_data = {}
            for name in raw_data.keys():
                scaled_data[name] = raw_data[name] / vmax
                
            # Save as JSON
            with open(data_path+'/'+fisher_dir+'/'+filename.split('.')[0]+'_MinMaxScaled.json', "w") as outfile:
                json.dump({k: scaled_data[k].tolist() for k, v in scaled_data.items()}, outfile)
                
            scaled_data_step.append(scaled_data)
            
        # Calculate average over all runs
        total_scaled_data = None
        n = 0
        for run_scaled_data in scaled_data_step:
            if total_scaled_data is None:
                total_scaled_data = run_scaled_data
            else:
                for (total_n, total_v), (run_n, run_v) in zip(total_scaled_data.items(), run_scaled_data.items()):
                    assert (total_n == run_n)
                    total_scaled_data[total_n] = np.array(total_v) + np.array(run_v)
            n += 1
        for name in total_scaled_data.keys():
            total_scaled_data[name] = total_scaled_data[name] / n
            
        # Save as JSON
        with open(f"{data_path}/average-scaled-fisher-importance_step-{step}.json", "w") as outfile:
            json.dump({k: total_scaled_data[k].tolist() for k, v in total_scaled_data.items()}, outfile)

def performance_matrix(data_path, train_scheme, task_interval, performance_metric="reward"):

    if train_scheme == "sequential-inc":
        scenarios = ["perpendicular", "diagonal-25", "diagonal-50", "parallel"]
    elif train_scheme == "sequential-dec":
        scenarios = ["parallel", "diagonal-50", "diagonal-25", "perpendicular"]

    # Read data
    df_list = []
    for s in scenarios:
        df = pd.read_csv(f"{data_path}/{s}_{performance_metric}_mean.csv")

        # Get indices of task switches
        df_filtered = df[df["idx"] % task_interval == 0]
        df_filtered = pd.concat([df_filtered, df.tail(1)]) # Get last entry incase last evaluation wasn't exactly 2M steps
        
        index_list = df_filtered["idx"].drop_duplicates().index

        df_list.append(df.loc[df.index[index_list]])
        columns = df_filtered.columns[1:].tolist()

    # Construct matrix and average across runs
    matrix_avg = None
    for run in columns:
        cols = []
        for df in df_list:
            cols.append(df[run])
        matrix = pd.concat(cols, axis="columns").to_numpy()
        matrix = matrix.transpose()

        if matrix_avg is None:
            matrix_avg = matrix
        else:
            matrix_avg += matrix

    matrix_avg = matrix_avg / len(columns)

    np.savetxt(f"{data_path}/average_performance_matrix-{performance_metric}.csv", matrix_avg, delimiter=',', fmt='%.5f')
        
# Functions for plotting
# -----------------------------------------------------------------------------

# Format plot labels
# Code from https://stackoverflow.com/questions/59969492/how-to-print-10k-20k-1m-in-the-xlabel-of-matplotlib-plot
def format_func(value, tick_number=None):
    num_thousands = 0 if abs(value) < 1000 else math.floor (math.log10(abs(value))/3)
    value = round(value / 1000**num_thousands, 2)
    return f'{value:g}'+' KMGTPEZY'[num_thousands]

def plot_learning_curve(plot_path, data_paths, max_steps=2_000_000, task_interval=30_000,
                        metrics=["reward", "success", "crashed", "truncated"],
                        scenarios = ["perpendicular", "diagonal-25", "diagonal-50", "parallel"],
                        agent_list = ["ideal", "random", "sac", "ppo", "drama", "ppo-ewc"],
                        plt_kwargs = {"ideal"  : {"color" : mcolors.TABLEAU_COLORS["tab:green"]},
                                      "random" : {"color" : mcolors.TABLEAU_COLORS["tab:red"]}, 
                                      "sac"    : {"color" : mcolors.TABLEAU_COLORS["tab:blue"]}, 
                                      "ppo"    : {"color" : mcolors.TABLEAU_COLORS["tab:orange"]}, 
                                      "drama"  : {"color" : mcolors.TABLEAU_COLORS["tab:purple"]},
                                      "ppo-ewc": {"color" : mcolors.TABLEAU_COLORS["tab:brown"]},
                                     },
                       subplot_height=3, subplot_width=12, plot_envs=None, env_np_path=None, save_format="png",
                       plt_std=True):
    """
    Aggregate metrics for learning curve across runs and plot line graphs.

    Parameters
    ----------
    plot_path : str
        Path to directory for storing the plot.
    data_paths : dict
        Dictionary of paths to data directories. Keys should be name of the agent
        (ideal, random, SAC, PPO, Drama). Values should be paths to corresponding
        data directories

    Returns
    -------
    None.

    """
    plt_label = ["i.", "ii.", "iii.", "iv.", "v."]
    
    # Create missing directories
    if not os.path.exists(plot_path):
        print(f"Creating output directory {plot_path}")
        os.makedirs(plot_path)
        
    for m in metrics:            
        # Set up plot
        ncols = 1
        nrows = len(scenarios)
        
        # Include additional subplot for scenarios in sequential
        if plot_envs is not None:
            total_width = ncols*subplot_width
            total_height = nrows*subplot_height+(subplot_width/len(scenarios))
            gridspec_kw={'height_ratios': [subplot_width/len(scenarios)]+[subplot_height]*nrows}
            
            fig, ax = plt.subplots(ncols=ncols, nrows=nrows+1, figsize=(total_width, total_height),gridspec_kw=gridspec_kw)

            order = None
            if plot_envs=="sequential-inc":
                order = ["perpendicular", "diagonal-25", "diagonal-50", "parallel"]
            elif plot_envs=="sequential-dec":
                order = ["parallel", "diagonal-50", "diagonal-25", "perpendicular"]

            img = None
            for t in order:
                if img is None:
                    img = np.load(f"{env_np_path}/{t}_parking.npy")
                else:
                    img = np.concatenate((img, np.load(f"{env_np_path}/{t}_parking.npy")), axis=1)

            # Plot images of tasks
            ax[0].imshow(img, extent=[0,2_000_000-1,500_000,0],aspect="auto")

            # Format y-axis
            ax[0].set_ylabel("current task")
            ax[0].tick_params(axis='y',        # changes apply to the x-axis
                              which='both',    # both major and minor ticks are affected
                              left=False,      # ticks along the left edge are off
                              labelleft=False)

            # Format x-axis
            ax[0].xaxis.set_major_formatter(plt.FuncFormatter(format_func))
            ax[0].set_xlabel("Environment steps")
            

            # Add vertical lines indicating task switch
            pos = task_interval
            while pos < 2_000_000:
                ax[0].axvline(x=pos, color="white", linestyle='dotted')
                pos += task_interval

            pos = 250_000
            trans = ax[0].get_xaxis_transform() # x in data untis, y in axes fraction
            for t in order:
                ax[0].annotate(t, xy=(pos, 1.05), xycoords=trans,ha='center',fontsize='small')
                pos += task_interval
            
        else:
            fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(ncols*subplot_width, nrows*subplot_height))
        
        # Init axis limits for all subplots
        xmin = 0
        xmax = max_steps
        ymin = 0
        ymax = 0
                
        for i in range(nrows):
            # Check if there is more than 1 subplot
            if nrows > 1 and plot_envs is None:
                ax_i = ax[i]
            elif nrows > 1 and plot_envs is not None:
                ax_i = ax[i+1]
            else:
                ax_i = ax

            
            # Plot metrics for each type of agent
            for agent in agent_list:
                    
                x = None
                y = None
                std = None
                    
                if agent == "ideal":
                    # Get ideal value for the metric
                    if m == "reward":
                        ideal_value = 0 # Rewards are negative
                    elif m == "success":
                        ideal_value = 1
                    elif m in ["crashed", "truncated"]:
                        ideal_value = 0
                            
                    x = [i for i in range(max_steps)]
                    y = np.ones(max_steps)*ideal_value
                    std = np.zeros(max_steps)
                else:
                    if agent in data_paths.keys():
                        # Read data
                        path = data_paths[agent] + f"/{scenarios[i]}_{m}_mean.csv"
                        df = pd.read_csv(path)
                            
                        # Calculate mean and std
                        x = df["idx"]
                        y = df[df.columns[1:]].mean(axis="columns")
                        std = df[df.columns[1:]].std(axis="columns")
                    
                # Data was read
                if x is not None and y is not None and std is not None:
                        
                    # Update axis limits
                    xmax = min([xmax, max(x)])
                    ymin = min([ymin, min(y)])
                    ymax = max([ymax, max(y)])
                        
                    # Plot mean and std
                    ax_i.plot(x, y, label=agent, **plt_kwargs[agent])

                    if plt_std:
                        ax_i.fill_between(x, y-std, y+std, alpha=0.3, **plt_kwargs[agent])
                        
            ax_i.set_ylabel(f"{scenarios[i]}\n{m}")
            if nrows > 1:
                ax_i.annotate(plt_label[i], xy=(-0.2, 0.95), xycoords="axes fraction",ha='center',fontsize='large')
                
            # Format y axis as percentages and set range to 0%-100%
            if m in ["success", "crashed", "truncated"]:
                ax_i.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
                ymin = 0
                ymax = 1
                
            # Add legend to top plot
            if i == 0:
                ax_i.legend(loc='upper left', bbox_to_anchor=(1, 1))
                
        # Use same axis ranges across all plots
        for i in range(nrows):
            # Check if there is more than 1 subplot
            if nrows > 1 and plot_envs is None:
                ax_i = ax[i]
            elif nrows > 1 and plot_envs is not None:
                ax_i = ax[i+1]
            else:
                ax_i = ax
                
            ybuff = 0.1*(ymax-ymin)
            ax_i.set_xlim((xmin, xmax))
            ax_i.set_ylim((ymin-ybuff, ymax+ybuff))

            # Format x-axis numbers
            ax_i.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
            ax_i.set_xlabel("Environment steps")
            
            # Add vertical lines indicating task switch
            pos = task_interval
            while pos < xmax:
                ax_i.axvline(x=pos, color="black", linestyle='dotted')
                pos += task_interval
        
        #fig.suptitle(f"Average {m}")
                
        fig.tight_layout()
        fig.savefig(f"{plot_path}/learning_curve_{m}.{save_format}", format=save_format)

    plt.close('all')
        
def plot_AFS_heatmap(path, fisher_type="average-fisher-sensitivity", scale_all=False, save_format="png"):
    """
    fisher_type : "average-fisher-sensitivity" or "average-scaled-fisher-importance"

    scale_all : whether to min-max scale across the whole network or per layer
    """
    data_path = path + "/data"
    plot_path = path + "/plots"

    formatter = mtick.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))

    for fisher_file in [x for x in os.listdir(data_path) if x.startswith(fisher_type)]:
        
        step = int(fisher_file.split('-')[-1].split('.')[0])
        
        with open(data_path+'/'+fisher_file) as f:
            data = json.load(f)

        # Create plots for each layer
        layer_names = [x.replace(".weight", "").replace(".bias", "") for x in data.keys()]
        layer_names = np.unique(layer_names).tolist()

        # Get maximum values for whole network
        vmax = 0
        if scale_all:
            for layer in layer_names:
                if f"{layer}.weight" in data.keys() and f"{layer}.bias" in data.keys():
                    for name in [f"{layer}.weight", f"{layer}.bias"]:
                        data[name] = np.array(data[name])
                        vmax = max(vmax, np.max(data[name]))
                else:
                    data[layer] = np.array(data[layer])
                    vmax = max(vmax, np.max(data[layer]))

        for layer in layer_names:
            if not scale_all:
                vmax = 0
                
            if f"{layer}.weight" in data.keys() and f"{layer}.bias" in data.keys():
                for name in [f"{layer}.weight", f"{layer}.bias"]:
                    data[name] = np.array(data[name])
                    vmax = max(vmax, np.max(data[name]))
                    if len(data[name].shape) == 1:
                            data[name] = data[name].reshape((data[name].shape[0],1))
                    
                fig, ax = plt.subplots(ncols = 2, figsize=(7, 5), 
                                           gridspec_kw=dict(width_ratios=[5,2]))
                
                i=0
                for suffix in ["weight", "bias"]:
                    # Get matrix shape
                    data_rows, data_cols = data[f"{layer}.{suffix}"].shape

                    # Plot heatmap
                    im = ax[i].imshow(data[f"{layer}.{suffix}"], vmin=0, vmax=vmax,aspect="auto", cmap='magma', extent=(0,data_cols,data_rows,0))
                    ax[i].set_title(suffix, y=-0.11)

                    # Calculate minor axis ticks
                    ax[i].set_xticks(np.arange(1, data_cols+1)-.5, minor=True)
                    ax[i].set_yticks(np.arange(1, data_rows+1)-.5, minor=True)

                    # Calculate major x-axis ticks
                    step_func = lambda x : (1 if x < 5 else (3 if x < 15 else (10 if x < 75 else 25)))
                    tick_array = np.concatenate(([1], np.arange(step_func(data_cols), data_cols+1, step_func(data_cols))))
                    ax[i].set_xticks(tick_array-.5, labels=tick_array)
                    
                    # Calculate major y-axis ticks
                    tick_array = np.concatenate(([1], np.arange(step_func(data_rows), data_rows+1, step_func(data_rows))))
                    ax[i].set_yticks(tick_array-.5, labels=tick_array)
                    
                    if suffix == "weight":
                        # Let the horizontal axes labeling appear on top.
                        ax[i].tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

                        # Draw minor ticks
                        ax[i].tick_params(which="minor", top=True, left=True, bottom=False, )
                        
                    else:
                        # Hide axis labels for bias
                        ax[i].tick_params(which="both", left=False, bottom=False, labelleft=False, labelbottom=False)

                        # Create colorbar
                        cbar = fig.colorbar(im, ax=ax[i], format="%.2E", pad=0.7, fraction=0.5)
                        cbar.ax.set_ylabel("Average Fisher Sensitivity", rotation=90, va="bottom")
                        cbar.ax.yaxis.set_label_position('left')
    
                    i+=1
            else:
                data[layer] = np.array(data[layer])
                vmax = max(vmax, np.max(data[layer]))
                if len(data[layer].shape) == 1:
                    data[layer] = data[layer].reshape((data[layer].shape[0],1))
                    
                fig, ax = plt.subplots(ncols = 1, figsize=(6, 5))
                
                # Get matrix shape
                data_rows, data_cols = data[f"{layer}"].shape

                # Plot heatmap
                im = ax.imshow(data[f"{layer}"], vmin=0, vmax=vmax,aspect="auto", cmap='magma', extent=(0,data_cols,data_rows,0))

                # Calculate minor axis ticks
                ax.set_xticks(np.arange(1, data_cols+1)-.5, minor=True)
                ax.set_yticks(np.arange(1, data_rows+1)-.5, minor=True)

                # Calculate major x-axis ticks
                step_func = lambda x : (1 if x < 5 else (3 if x < 15 else (10 if x < 75 else 25)))
                tick_array = np.concatenate(([1], np.arange(step_func(data_cols), data_cols+1, step_func(data_cols))))
                ax.set_xticks(tick_array-.5, labels=tick_array)
                    
                # Calculate major y-axis ticks
                tick_array = np.concatenate(([1], np.arange(step_func(data_rows), data_rows+1, step_func(data_rows))))
                ax.set_yticks(tick_array-.5, labels=tick_array)

                # Let the horizontal axes labeling appear on top.
                ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

                # Draw minor ticks
                ax.tick_params(which="minor", top=True, left=True, bottom=False, )

                # Create colorbar
                cbar = fig.colorbar(im, ax=ax, format="%.2E", pad=0.2, fraction=0.5)
                cbar.ax.set_ylabel("Average Fisher Sensitivity", rotation=90, va="bottom")
                cbar.ax.yaxis.set_label_position('left')

            fig.suptitle(f"Layer: {layer}\nStep: {step}")
            
            fig.tight_layout()
            fig.subplots_adjust(wspace=0.2, hspace=0)

            if scale_all:
                fig.savefig(f"{plot_path}/AFS_{layer}_step-{step}.{save_format}", format=save_format)
            else:
                fig.savefig(f"{plot_path}/AFS_layerwise_{layer}_step-{step}.{save_format}", format=save_format)

    plt.close('all')