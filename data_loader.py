import h5py
import numpy as np
import tensorflow as tf

class RadioMLDataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.modulations = [
            'OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK',
            '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM',
            '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC',
            'FM', 'GMSK', 'OQPSK'
        ]

    def normalize(self, x):
        """Z-Score standardization: ensures mean 0 and unit variance."""
        # 1. Scrub NaNs and Infs
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 2. Standardize each sample individually
        # Calculate mean and std across the time axis (1024 samples)
        mean = np.mean(x, axis=1, keepdims=True) # [Batch, 1, 2]
        std = np.std(x, axis=1, keepdims=True)   # [Batch, 1, 2]
        
        # 3. Apply formula: (x - mean) / std
        return (x - mean) / (std + 1e-8)

    def get_generator(self, indices, batch_size=64):
        """Turbocharged generator: reads contiguous chunks for maximum speed."""
        count = 0
        chunk_size = 4096 # Read 4096 samples at a time (Fast contiguous I/O)
        
        with h5py.File(self.file_path, 'r') as f:
            X_ds = f['X']
            Y_ds = f['Y']
            
            print(f"Opal Vanguard: Turbocharged Pipe Primed.")
            while True:
                # To maintain randomness, we shuffle the order of chunks
                np.random.shuffle(indices)
                
                for i in range(0, len(indices), chunk_size):
                    # Get a chunk of indices and sort them for HDF5 speed
                    chunk_idx = sorted(indices[i:i+chunk_size])
                    
                    # Contiguous-ish read
                    X_chunk = X_ds[chunk_idx]
                    Y_chunk = Y_ds[chunk_idx]
                    
                    # Shuffle this chunk in RAM to maintain high-quality randomness
                    p = np.random.permutation(len(X_chunk))
                    X_chunk = X_chunk[p]
                    Y_chunk = Y_chunk[p]
                    
                    # Yield batches from this fast RAM-resident chunk
                    for j in range(0, len(X_chunk), batch_size):
                        X_batch = X_chunk[j:j+batch_size]
                        Y_batch = Y_chunk[j:j+batch_size]
                        
                        if len(X_batch) < batch_size:
                            continue
                            
                        X_batch = self.normalize(X_batch)
                        
                        count += 1
                        if count % 10 == 0:
                            print(f".", end="", flush=True)
                        
                        yield X_batch, Y_batch

    def get_train_val_indices(self, test_size=0.2, seed=42):
        """Returns indices for training and validation splits without loading X/Y."""
        with h5py.File(self.file_path, 'r') as f:
            n_samples = f['X'].shape[0]
        
        np.random.seed(seed)
        indices = np.random.permutation(n_samples)
        split = int(n_samples * (1 - test_size))
        return indices[:split], indices[split:]

if __name__ == "__main__":
    # Placeholder for local testing
    print("Opal Vanguard: Data Loader initialized. Awaiting 2018.01A dataset.")
