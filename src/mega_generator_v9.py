import os
import sys
import h5py
import numpy as np
import time
import gc
from tqdm import tqdm
from torchsig.utils.defaults import default_dataset
from torchsig.signals.signal_lists import SIGNALS_SHARED_LIST

# Configuration
NUM_CLASSES = 57
SAMPLES_PER_CLASS_CLEAN = 10000  # 570k samples
SAMPLES_PER_CLASS_HARD = 10000   # 570k samples
TOTAL_SAMPLES = (SAMPLES_PER_CLASS_CLEAN + SAMPLES_PER_CLASS_HARD) * NUM_CLASSES
OUTPUT_FILE = "data/VDF_MEGA_SYNTHETIC_1M.h5"
IQ_LENGTH = 1024

def generate_dataset():
    print(f"Opal Vanguard: Initiating Stabilized Mega-TorchSig Factory")
    print(f"Target: 1.14 Million Samples | 57 Classes | Level 0 & 2.")
    sys.stdout.flush()
    
    os.makedirs("data", exist_ok=True)
    class_to_idx = {name: i for i, name in enumerate(SIGNALS_SHARED_LIST)}

    # Fresh write for the overnight marathon
    with h5py.File(OUTPUT_FILE, 'w') as f:
        x_ds = f.create_dataset('X', (TOTAL_SAMPLES, IQ_LENGTH, 2), dtype='float32')
        y_ds = f.create_dataset('Y', (TOTAL_SAMPLES, NUM_CLASSES), dtype='float32')
        current_idx = 0

        for stage_name, samples_per_class, level in [("CLEAN", SAMPLES_PER_CLASS_CLEAN, 0), 
                                                     ("HARDENED", SAMPLES_PER_CLASS_HARD, 2)]:
            print(f"\n--- Stage: {stage_name} (Level {level}) ---")
            sys.stdout.flush()
            
            for class_idx, class_name in enumerate(SIGNALS_SHARED_LIST):
                print(f" Generating {class_name} ({class_idx+1}/{NUM_CLASSES})...")
                sys.stdout.flush()
                
                dataset = default_dataset(
                    impairment_level=level,
                    num_iq_samples_dataset=IQ_LENGTH,
                    num_samples_dataset=samples_per_class,
                    use_class_list=[class_name]
                )
                
                it = iter(dataset)
                pbar = tqdm(total=samples_per_class, desc=f" {class_name}", unit="samples")
                
                for _ in range(samples_per_class):
                    try:
                        try:
                            data, _ = next(it)
                        except ValueError:
                            data = np.zeros(IQ_LENGTH, dtype=np.complex64)
                        
                        if len(data) > IQ_LENGTH: data = data[:IQ_LENGTH]
                        elif len(data) < IQ_LENGTH: data = np.pad(data, (0, IQ_LENGTH - len(data)), 'constant')
                        
                        iq = np.stack([np.real(data), np.imag(data)], axis=-1)
                        iq = iq / (1.0 + np.abs(iq)) # Soft-Clip
                        
                        x_ds[current_idx] = iq
                        
                        one_hot = np.zeros(NUM_CLASSES)
                        one_hot[class_idx] = 1.0 
                        y_ds[current_idx] = one_hot
                        
                        current_idx += 1
                        pbar.update(1)
                    except StopIteration:
                        break
                pbar.close()
                
                # STABILITY HOOKS:
                f.flush()
                del dataset
                del it
                gc.collect()
                time.sleep(5) # Shorter cooldown for large run
                sys.stdout.flush()

    print(f"\nMission Success: Full Mega-Dataset saved to {OUTPUT_FILE}")
    sys.stdout.flush()

if __name__ == "__main__":
    generate_dataset()
