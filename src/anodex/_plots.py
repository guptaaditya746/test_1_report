
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_timeline(scores, selected_idx, out_file):
    """Plots anomaly scores over time and highlights the selected index."""
    plt.figure(figsize=(12, 4))
    plt.plot(scores, label='Anomaly Score')
    plt.axvline(selected_idx, color='r', linestyle='--', label=f'Selected Anomaly (idx={selected_idx})')
    plt.xlabel("Instance Index")
    plt.ylabel("Anomaly Score")
    plt.title("Anomaly Scores on Test Set")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()

def plot_cf_overlay(x, x_cf, feature_names, out_dir):
    """Plots the original and counterfactual time series for each feature."""
    if x.ndim == 1:
        x = x.reshape(1, -1)
        x_cf = x_cf.reshape(1, -1)
    
    n_features = x.shape[-1]
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(n_features)]

    for i in range(n_features):
        plt.figure(figsize=(12, 4))
        plt.plot(x[:, i], label='Original')
        plt.plot(x_cf[:, i], label='Counterfactual', linestyle='--')
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.title(f"Counterfactual Overlay: {feature_names[i]}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(out_dir / f"cf_overlay_{i}.png", bbox_inches='tight')
        plt.close()

def plot_opt_trace(history_df, out_file):
    """Plots the optimization trace from the counterfactual generation."""
    fig, ax1 = plt.subplots(figsize=(12, 4))

    ax1.plot(history_df['objective'], 'b-', label='Objective')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Objective', color='b')
    ax1.tick_params('y', colors='b')

    ax2 = ax1.twinx()
    ax2.plot(history_df['p_anom'], 'r-', label='P(anomaly)')
    ax2.set_ylabel('P(anomaly)', color='r')
    ax2.tick_params('y', colors='r')

    plt.title("Optimization Trace")
    plt.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()
