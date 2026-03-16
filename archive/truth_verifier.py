import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

def verify_truth(file_path="VDF_INDUSTRIAL_TOTAL.h5"):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    print(f"--- Opal Vanguard: Truth Verification for {file_path} ---")
    
    with h5py.File(file_path, 'r') as f:
        X = f['X']
        Y = f['Y']
        Z = f['Z']
        
        num_samples = X.shape[0]
        modulations = [
            '32PSK', '16APSK', '32QAM', 'FM', 'GMSK', '32APSK', 'OQPSK', '8ASK',
            'BPSK', '8PSK', 'AM-SSB-SC', '4ASK', '16PSK', '64APSK', '128QAM',
            '128APSK', 'AM-DSB-SC', 'AM-SSB-WC', '64QAM', 'QPSK', '256QAM',
            'AM-DSB-WC', 'OOK', '16QAM'
        ]

        # Select 4 distinct modulations to visualize
        unique_classes = np.unique(np.argmax(Y[:], axis=1))
        print(f"Detected Classes in file: {[modulations[i] for i in unique_classes]}")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes = axes.flatten()

        for i, class_idx in enumerate(unique_classes[:4]):
            # Find the first sample of this class
            idx = np.where(np.argmax(Y[:], axis=1) == class_idx)[0][0]
            
            sample = X[idx]
            label = modulations[class_idx]
            metadata = Z[idx]
            
            # Plot Constellation
            axes[i].scatter(sample[:, 0], sample[:, 1], alpha=0.5, s=1, color='#00ff41')
            axes[i].set_title(f"Class: {label}\n(SNR: {metadata[0]}dB, Jam: {'YES' if metadata[4]==1 else 'NO'})")
            axes[i].set_xlim([-1.1, 1.1])
            axes[i].set_ylim([-1.1, 1.1])
            axes[i].grid(True, alpha=0.2)
            axes[i].set_aspect('equal')

        plt.tight_layout()
        plt.savefig('truth_verification.png')
        print("\nVerification Chart Saved: truth_verification.png")
        
        # Power Check
        print("\n--- Power Audit ---")
        clean_idx = np.where(Z[:, 4] == 0)[0][:100]
        jam_idx = np.where(Z[:, 4] == 1)[0][:100]
        
        if len(clean_idx) > 0:
            p_clean = np.mean(np.abs(X[clean_idx])**2)
            print(f"Mean Power (Clean): {p_clean:.6f}")
        if len(jam_idx) > 0:
            p_jam = np.mean(np.abs(X[jam_idx])**2)
            print(f"Mean Power (Jammed): {p_jam:.6f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default="VDF_INDUSTRIAL_TOTAL.h5", help='Path to the .h5 file to verify')
    args = parser.parse_args()
    
    verify_truth(args.file)
