import os
import h5py
import numpy as np
from tqdm import tqdm
from torchsig.utils.defaults import default_dataset
from torchsig.signals.signal_lists import SIGNALS_SHARED_LIST

def generate_torchsig_v2_hdf5(num_samples=10000, output_file="torchsig_v2_industrial.h5"):
    print(f"Generating TorchSig V2 dataset with {num_samples} samples...")
    
    # 1. Define the class list (all 53 signals in TorchSig)
    classes = SIGNALS_SHARED_LIST
    class_to_idx = {name: i for i, name in enumerate(classes)}
    num_classes = len(classes)
    
    # 2. Instantiate the dataset
    # We use impairment_level=1 (Cabled reference)
    dataset = default_dataset(
        impairment_level=1,
        num_iq_samples_dataset=1024,
        signal_generators="all",
        target_labels=["class_name"],
        num_signals_min=1,
        num_signals_max=1,
        signal_duration_in_samples_min=1024,
        signal_duration_in_samples_max=1024,
        bandwidth_min=0.2 * 10000000, # Strictly less than sample_rate/2
        bandwidth_max=0.4 * 10000000
    )
    
    # 3. Setup output file
    if os.path.exists(output_file): os.remove(output_file)
    
    with h5py.File(output_file, 'w') as f:
        X_ds = f.create_dataset('X', shape=(num_samples, 1024, 2), dtype=np.float32)
        Y_ds = f.create_dataset('Y', shape=(num_samples, num_classes), dtype=np.float32)
        
        # 4. Generation Loop
        # TorchSigIterableDataset returns (data, class_name)
        it = iter(dataset)
        i = 0
        pbar = tqdm(total=num_samples)
        while i < num_samples:
            try:
                data, class_name = next(it)
                
                # data is complex64 (1024,)
                # Convert to [I, Q] (1024, 2)
                iq = np.stack([np.real(data), np.imag(data)], axis=-1)
                
                # Soft-clip normalization to match V7 standard
                iq = iq / (1.0 + np.abs(iq))
                
                X_ds[i] = iq
                
                # Label
                y_onehot = np.zeros(num_classes, dtype=np.float32)
                if class_name in class_to_idx:
                    y_onehot[class_to_idx[class_name]] = 1.0
                Y_ds[i] = y_onehot
                
                i += 1
                pbar.update(1)
            except Exception as e:
                # Skip samples that cause internal TorchSig math errors
                continue
        pbar.close()
            
    print(f"✓ Successfully generated {output_file}")
    print(f"Total Classes: {num_classes}")

if __name__ == "__main__":
    generate_torchsig_v2_hdf5(num_samples=100000)
