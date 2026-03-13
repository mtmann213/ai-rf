import h5py
import numpy as np
import tensorflow as tf

class RadioMLDataLoader:
    """
    Data Loader for RadioML 2018.01A Dataset.
    Dataset contains:
    - X: [N, 1024, 2] I/Q samples
    - Y: [N, 24] One-hot encoded labels
    - Z: [N, 1] SNR values
    """
    def __init__(self, file_path):
        self.file_path = file_path
        self.modulations = [
            'OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK',
            '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM',
            '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC',
            'FM', 'GMSK', 'OQPSK'
        ]

    def load_data(self):
        """Loads the raw X, Y, Z data from the HDF5 file."""
        try:
            with h5py.File(self.file_path, 'r') as f:
                X = f['X'][:]  # [N, 1024, 2]
                Y = f['Y'][:]  # [N, 24]
                Z = f['Z'][:]  # [N, 1]
            return X, Y, Z
        except FileNotFoundError:
            print(f"Error: Dataset not found at {self.file_path}")
            return None, None, None

    def get_train_test_split(self, test_size=0.2, seed=42):
        """Prepares a split for training and evaluation."""
        X, Y, Z = self.load_data()
        if X is None: return None
        
        n_samples = X.shape[0]
        np.random.seed(seed)
        indices = np.random.permutation(n_samples)
        
        split = int(n_samples * (1 - test_size))
        train_idx, test_idx = indices[:split], indices[split:]
        
        return (X[train_idx], Y[train_idx], Z[train_idx]), (X[test_idx], Y[test_idx], Z[test_idx])

if __name__ == "__main__":
    # Placeholder for local testing
    print("Opal Vanguard: Data Loader initialized. Awaiting 2018.01A dataset.")
