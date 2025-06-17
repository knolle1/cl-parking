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
        for filename in [x for x in os.listdir(data_path+'/'+fisher_dir) if "_AFS" not in x and x.endswith(".json")]:
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
                total_afs = afs
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
        
# Functions for plotting
# -----------------------------------------------------------------------------

# Define colour map for consistency accross plots
colours = {"ideal" : mcolors.TABLEAU_COLORS["tab:green"],
           "random" : mcolors.TABLEAU_COLORS["tab:red"],
           "sac" : mcolors.TABLEAU_COLORS["tab:blue"],
           "ppo" : mcolors.TABLEAU_COLORS["tab:orange"],
           "drama" : mcolors.TABLEAU_COLORS["tab:purple"],
           }

# Format plot labels
# Code from https://stackoverflow.com/questions/59969492/how-to-print-10k-20k-1m-in-the-xlabel-of-matplotlib-plot
def format_func(value, tick_number=None):
    num_thousands = 0 if abs(value) < 1000 else math.floor (math.log10(abs(value))/3)
    value = round(value / 1000**num_thousands, 2)
    return f'{value:g}'+' KMGTPEZY'[num_thousands]

def plot_learning_curve(plot_path, data_paths, max_steps=2_000_000, task_interval=30_000,
                        metrics=["reward", "success", "crashed", "truncated"],
                        scenarios = ["perpendicular", "diagonal-25", "diagonal-50", "parallel"]):
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
    
    # Create missing directories
    if not os.path.exists(plot_path):
        print(f"Creating output directory {plot_path}")
        os.makedirs(plot_path)
        
    for m in metrics:
        # Set up plot
        ncols = 1
        nrows = len(scenarios)
        fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(ncols*10, nrows*3))
        
        # Init axis limits for all subplots
        xmin = 0
        xmax = max_steps
        ymin = 0
        ymax = 0
                
        for i in range(nrows):
            # Check if there is more than 1 subplot
            if nrows > 1:
                ax_i = ax[i]
            else:
                ax_i = ax
                
            # Plot metrics for each type of agent
            for agent in ["ideal", "random", "sac", "ppo", "drama"]:
                
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
                    ax_i.plot(x, y, label=agent, color=colours[agent])
                    ax_i.fill_between(x, y-std, y+std, alpha=0.3, color=colours[agent])
                    
            ax_i.set_ylabel(f"{scenarios[i]}")
            ax_i.set_xlabel("Environment steps")
            
            # Format y axis as percentages and set range to 0%-100%
            if m in ["success", "crashed", "truncated"]:
                ax_i.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
                ymin = 0
                ymax = 1
            
            # Format x-axis numbers
            ax_i.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
                
            # Add legend to top plot
            if i == 0:
                ax_i.legend(loc='upper left', bbox_to_anchor=(1, 1))
                
        # Use same axis ranges across all plots
        for i in range(nrows):
            # Check if there is more than 1 subplot
            if nrows > 1:
                ax_i = ax[i]
            else:
                ax_i = ax
                
            ybuff = 0.1*(ymax-ymin)
            ax_i.set_xlim((xmin, xmax))
            ax_i.set_ylim((ymin-ybuff, ymax+ybuff))
            
            # Add vertical lines indicating task switch
            pos = task_interval
            while pos < xmax:
                ax_i.axvline(x=pos, color="black", linestyle='dotted')
                pos += task_interval
        
        fig.suptitle(f"Average {m}")
                
        fig.tight_layout()
        fig.savefig(f"{plot_path}/learning_curve_{m}.png")
        
def plot_AFS_heatmap(path):
    data_path = path + "/data"
    plot_path = path + "/plots"

    formatter = mtick.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))

    for fisher_file in [x for x in os.listdir(data_path) if x.startswith("average-fisher-sensitivity")]:
        
        step = int(fisher_file.split('-')[-1].split('.')[0])
        
        with open(data_path+'/'+fisher_file) as f:
            data = json.load(f)

        # Create plots for each layer
        layer_names = [x.replace(".weight", "").replace(".bias", "") for x in data.keys()]
        layer_names = np.unique(layer_names).tolist()
        
        for layer in layer_names:
            
            if f"{layer}.weight" in data.keys() and f"{layer}.bias" in data.keys():
                vmax = 0
                for name in [f"{layer}.weight", f"{layer}.bias"]:
                    data[name] = np.array(data[name])
                    vmax = max(vmax, np.max(data[name]))
                    if len(data[name].shape) == 1:
                            data[name] = data[name].reshape((data[name].shape[0],1))
                    
                fig, ax = plt.subplots(ncols = 3, figsize=(7, 5), 
                                           gridspec_kw=dict(width_ratios=[10,2,1]))
                
                i=0
                for suffix in ["weight", "bias"]:
                    sns.heatmap(data[f"{layer}.{suffix}"], ax=ax[i],  cbar=False, 
                                xticklabels=False, yticklabels=False)
                    ax[i].set_title(suffix, y=-0.1)
                    i+=1
            else:
                vmax = 0
                data[layer] = np.array(data[layer])
                vmax = max(vmax, np.max(data[layer]))
                if len(data[layer].shape) == 1:
                    data[layer] = data[layer].reshape((data[layer].shape[0],1))
                    
                fig, ax = plt.subplots(ncols = 2, figsize=(7, 5), 
                                           gridspec_kw=dict(width_ratios=[12,1]))
                
                sns.heatmap(data[f"{layer}"], ax=ax[0],  cbar=False, 
                                xticklabels=False, yticklabels=False)
                #ax[0].set_title(suffix, y=-0.1)
                
            fig.suptitle(f"Layer: {layer}\nStep: {step}")
            fig.colorbar(ax[0].collections[0], cax=ax[-1], format="%.2E")
            fig.tight_layout()
            fig.subplots_adjust(wspace=0.1, hspace=0)
            
            fig.savefig(f"{plot_path}/AFS_{layer}_step-{step}.png")