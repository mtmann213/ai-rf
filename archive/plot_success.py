import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_success_only():
    if not os.path.exists('training_log_v7.csv'):
        print("Error: training_log_v7.csv not found.")
        return
        
    df = pd.read_csv('training_log_v7.csv')
    df = df[df['epoch'] != 'epoch'].apply(pd.to_numeric)
    
    # We only want the V7.6.1 rows. The first 7 rows in the CSV are from the V7.6.1 
    # run before the crash, and the rows after that are the resumed run.
    # Since we appended to the file, we just need to plot the whole thing sequentially.
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    plt.subplots_adjust(hspace=0.3)

    epochs = range(len(df['accuracy']))

    # Plot Accuracy
    ax1.plot(epochs, df['accuracy'], color='#00ff41', linewidth=3, label='Train Accuracy')
    ax1.plot(epochs, df['val_accuracy'], color='#ff00ff', marker='s', label='Val Accuracy')
    
    ax1.axhline(y=0.0416, color='white', linestyle='--', alpha=0.3, label='Random Guessing Baseline (4.16%)')
    ax1.set_title('Opal Vanguard: Event Horizon Validation (V7.6.1)', fontsize=14, color='white')
    ax1.set_ylabel('Accuracy', color='white')
    ax1.legend(facecolor='#1e1e1e', labelcolor='white')
    ax1.set_facecolor('#0a0a0a')
    ax1.grid(True, alpha=0.1)

    # Plot Loss
    ax2.plot(epochs, df['loss'], color='#008f11', linewidth=3, label='Train Loss')
    ax2.plot(epochs, df['val_loss'], color='#bc00bc', marker='s', label='Val Loss')
    ax2.set_title('Mathematical Descent (Loss)', fontsize=14, color='white')
    ax2.set_ylabel('Loss', color='white')
    ax2.set_xlabel('Epochs', color='white')
    ax2.legend(facecolor='#1e1e1e', labelcolor='white')
    ax2.set_facecolor('#0a0a0a')
    ax2.grid(True, alpha=0.1)

    fig.patch.set_facecolor('#0a0a0a')
    for ax in [ax1, ax2]:
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('#444444')

    plt.savefig('mission_success.png', facecolor='#0a0a0a')
    print("Success Plot Generated: mission_success.png")

if __name__ == "__main__":
    plot_success_only()
