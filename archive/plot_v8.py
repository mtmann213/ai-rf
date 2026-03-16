import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_v8():
    csv_path = 'training_log_v8.csv'
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Training must start first!")
        return

    df = pd.read_csv(csv_path)
    df = df[df['epoch'] != 'epoch'].apply(pd.to_numeric)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Plot Accuracy
    ax1.plot(df['epoch'], df['accuracy'], label='Train Accuracy', color='#00ff41', marker='o')
    ax1.plot(df['epoch'], df['val_accuracy'], label='Val Accuracy', color='#ff00ff', marker='s')
    ax1.set_title('Opal Vanguard V8: Universal Expansion Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot Loss
    ax2.plot(df['epoch'], df['loss'], label='Train Loss', color='#008f11', marker='o')
    ax2.plot(df['epoch'], df['val_loss'], label='Val Loss', color='#bc00bc', marker='s')
    ax2.set_title('Opal Vanguard V8: Universal Expansion Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('mission_progress_v8.png')
    print("Dashboard updated: mission_progress_v8.png")

if __name__ == "__main__":
    plot_v8()
