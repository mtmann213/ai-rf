import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_logs(csv_path='training_log_v7.csv', step_path='step_log_v7.csv'):
    # Prefer step-level logs if they exist for real-time visibility
    if os.path.exists(step_path) and os.path.getsize(step_path) > 100:
        print(f"Plotting real-time step data from {step_path}...")
        df = pd.read_csv(step_path)
        
        # Calculate continuous absolute steps across epochs
        # Whenever 'step' drops, it means a new epoch started
        epoch_offsets = df['step'].diff() < 0
        df['epoch_id'] = epoch_offsets.cumsum()
        
        # Find max step per epoch to add as offset
        max_steps = df.groupby('epoch_id')['step'].max().shift(1).fillna(0).cumsum()
        df['absolute_step'] = df['step'] + df['epoch_id'].map(max_steps)

        title_suffix = "(Continuous Steps)"
        x_axis = 'absolute_step'
    elif os.path.exists(csv_path):
        print(f"Plotting epoch data from {csv_path}...")
        df = pd.read_csv(csv_path)
        df = df[df['epoch'] != 'epoch']
        df = df.apply(pd.to_numeric)
        title_suffix = "(By Epoch)"
        x_axis = 'epoch'
    else:
        print("Error: No log files found yet. Mission training must begin first!")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Plot Accuracy
    ax1.plot(df[x_axis], df['accuracy'], label='Train Accuracy', color='#00ff41', linewidth=1)
    if 'val_accuracy' in df.columns:
        ax1.plot(df[x_axis], df['val_accuracy'], label='Val Accuracy', color='#ff00ff', marker='s')
    ax1.set_title(f'Opal Vanguard: Accuracy Trend {title_suffix}')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot Loss
    ax2.plot(df[x_axis], df['loss'], label='Train Loss', color='#008f11', linewidth=1)
    if 'val_loss' in df.columns:
        ax2.plot(df[x_axis], df['val_loss'], label='Val Loss', color='#bc00bc', marker='s')
    ax2.set_title(f'Opal Vanguard: Loss Trend {title_suffix}')
    ax2.set_xlabel(x_axis.capitalize())
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('mission_progress.png')
    print("Dashboard updated: mission_progress.png")

if __name__ == "__main__":
    plot_logs()
