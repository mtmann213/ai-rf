import h5py
import numpy as np
import tensorflow as tf
import sionna
from sionna.phy.mapping import BinarySource, Mapper, Constellation
from sionna.phy.channel import AWGN

# Configuration matching train_opal_vanguard.py
MODULATIONS = [
    'OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK',
    '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM',
    '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC',
    'FM', 'GMSK', 'OQPSK'
]
NUM_CLASSES = len(MODULATIONS)
SAMPLES_PER_MOD = 100 # Small batch for rapid testing
INPUT_LENGTH = 1024
FILENAME = "GOLD_XYZ_OSC.0001_1024.hdf5"

def get_sionna_constellation(mod_type):
    if 'QAM' in mod_type:
        m = int(mod_type.replace('QAM', ''))
        bits = int(np.log2(m))
        return Constellation("qam", num_bits_per_symbol=bits if bits % 2 == 0 else bits + 1)
    return Constellation("qam", num_bits_per_symbol=2) # Fallback

def generate_mini_dataset():
    print(f"Opal Vanguard: Generating synthetic {FILENAME}...")
    
    X = [] # Samples
    Y = [] # One-hot labels
    Z = [] # SNRs
    
    source = BinarySource()
    channel = AWGN()
    
    for snr in [0.0, 10.0, 20.0]: # Sample SNRs
        no = 10.0**(-snr/10.0)
        
        for idx, mod in enumerate(MODULATIONS):
            const = get_sionna_constellation(mod)
            mapper = Mapper(constellation=const)
            
            # Generate bits and map to symbols
            b = source([SAMPLES_PER_MOD, INPUT_LENGTH * const.num_bits_per_symbol])
            x = mapper(b)
            y = channel(x, no)
            
            # Convert to [I, Q] format [N, 1024, 2]
            y_iq = tf.stack([tf.math.real(y), tf.math.imag(y)], axis=-1).numpy()
            
            # Label (One-hot) [N, 24]
            label = np.zeros((SAMPLES_PER_MOD, NUM_CLASSES))
            label[:, idx] = 1
            
            X.append(y_iq)
            Y.append(label)
            Z.append(np.full((SAMPLES_PER_MOD, 1), snr))
            
    # Concatenate all data
    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y, axis=0)
    Z = np.concatenate(Z, axis=0)
    
    # Save to HDF5 (RadioML format)
    with h5py.File(FILENAME, 'w') as f:
        f.create_dataset('X', data=X)
        f.create_dataset('Y', data=Y)
        f.create_dataset('Z', data=Z)
    
    print(f"Successfully generated {FILENAME} with {X.shape[0]} samples.")

if __name__ == "__main__":
    generate_mini_dataset()
