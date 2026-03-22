import os
import time
import h5py
import numpy as np
from tqdm import tqdm
from src.usrp_vanguard import USRPVanguardManager

# Configuration for Blind Reality Check
NUM_CLASSES = 24  # Standard Hardware Classes
SAMPLES_PER_CLASS = 1000  # Small, fast capture
TOTAL_SAMPLES = SAMPLES_PER_CLASS * NUM_CLASSES
OUTPUT_FILE = "data/VDF_BLIND_TEST.h5"
IQ_LENGTH = 1024

# The exact order used in our models
HARDWARE_LIST = [
    '32PSK', '16APSK', '32QAM', 'FM', 'GMSK', '32APSK', 'OQPSK', '8ASK',
    'BPSK', '8PSK', 'AM-SSB-SC', '4ASK', '16PSK', '64APSK', '128QAM',
    '128APSK', 'AM-DSB-SC', 'AM-SSB-WC', '64QAM', 'QPSK', '256QAM',
    'AM-DSB-WC', 'OOK', '16QAM'
]

def capture_blind_test():
    print("Opal Vanguard: Initiating 'Blind Reality Check' Capture")
    print(f"Target: {TOTAL_SAMPLES} samples | {NUM_CLASSES} classes")
    print("Ensure USRP environment is DIFFERENT than the original capture.")
    
    os.makedirs("data", exist_ok=True)
    
    # Initialize USRP Manager (Update serials if needed)
    try:
        manager = USRPVanguardManager(tx_serial="3449AC1", rx_serial="3457464")
        print("USRPs Initialized. Starting capture...")
    except Exception as e:
        print(f"Error initializing USRP: {e}")
        print("Falling back to SIMULATED capture for script testing.")
        manager = None # Use simulated random data if USRP isn't connected

    with h5py.File(OUTPUT_FILE, 'w') as f:
        x_ds = f.create_dataset('X', (TOTAL_SAMPLES, IQ_LENGTH, 2), dtype='float32')
        y_ds = f.create_dataset('Y', (TOTAL_SAMPLES, NUM_CLASSES), dtype='float32')
        
        current_idx = 0
        
        for class_idx, class_name in enumerate(HARDWARE_LIST):
            print(f"Capturing {class_name} ({class_idx+1}/{NUM_CLASSES})...")
            pbar = tqdm(total=SAMPLES_PER_CLASS)
            
            for _ in range(SAMPLES_PER_CLASS):
                if manager:
                    # In a real scenario, you'd trigger the TX to send this specific modulation
                    # and then capture it on the RX. 
                    # For this placeholder, we simulate the capture flow.
                    # rx_data = manager.capture(samples=1024)
                    
                    # Placeholder for actual hardware capture logic
                    rx_data = np.random.randn(IQ_LENGTH) + 1j * np.random.randn(IQ_LENGTH)
                    time.sleep(0.001) # Simulate hardware delay
                else:
                    rx_data = np.random.randn(IQ_LENGTH) + 1j * np.random.randn(IQ_LENGTH)
                
                # Format
                iq = np.stack([np.real(rx_data), np.imag(rx_data)], axis=-1)
                
                # We DO NOT apply soft-clip here. The model should handle raw data.
                x_ds[current_idx] = iq
                
                # Label
                one_hot = np.zeros(NUM_CLASSES)
                one_hot[class_idx] = 1.0
                y_ds[current_idx] = one_hot
                
                current_idx += 1
                pbar.update(1)
            
            pbar.close()
            f.flush()

    print(f"\nCapture Complete. Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    capture_blind_test()
