import os
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import sionna
from sionna.utils import BinarySource, Mapper, Constellation
from sionna.channel import AWGN

# --- Project Opal Vanguard Configuration ---
# 24-modulation set from RadioML 2018.01A
MODULATIONS = [
    'OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK',
    '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM',
    '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC',
    'FM', 'GMSK', 'OQPSK'
]
NUM_CLASSES = len(MODULATIONS)
INPUT_LENGTH = 1024
INPUT_SHAPE = (INPUT_LENGTH, 2)
BATCH_SIZE = 128
EPOCHS = 10

# Blackwell GPU Optimization
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"Opal Vanguard: NVIDIA RTX Blackwell active. Found {len(gpus)} GPU(s).")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("Opal Vanguard: Warning - Running on CPU.")

class OpalVanguardModel(models.Model):
    """
    Neural Receiver for Modulation Classification.
    Architecture: Conv1D-based CNN with 'same' padding.
    """
    def __init__(self, num_classes):
        super(OpalVanguardModel, self).__init__()
        self.conv1 = layers.Conv1D(64, 7, padding='same', activation='relu', input_shape=INPUT_SHAPE)
        self.pool1 = layers.MaxPooling1D(2)
        self.conv2 = layers.Conv1D(128, 5, padding='same', activation='relu')
        self.pool2 = layers.MaxPooling1D(2)
        self.conv3 = layers.Conv1D(256, 3, padding='same', activation='relu')
        self.pool3 = layers.MaxPooling1D(2)
        self.conv4 = layers.Conv1D(256, 3, padding='same', activation='relu')
        self.pool4 = layers.MaxPooling1D(2)
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(512, activation='relu')
        self.dropout = layers.Dropout(0.4)
        self.out = layers.Dense(num_classes, activation='softmax')

    def call(self, x):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))
        x = self.pool4(self.conv4(x))
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x)
        return self.out(x)

def get_sionna_constellation(mod_type):
    """Maps RadioML labels to Sionna Constellations."""
    if 'QAM' in mod_type:
        m = int(mod_type.replace('QAM', ''))
        return Constellation("qam", num_bits_per_symbol=int(np.log2(m)))
    elif 'PSK' in mod_type:
        m = int(mod_type.replace('PSK', ''))
        # PSK mapping in Sionna varies; using QAM-like bit depth for boilerplate
        return Constellation("qam", num_bits_per_symbol=int(np.log2(m)))
    else:
        # Default to BPSK for analog/complex types not yet implemented in Sionna
        return Constellation("qam", num_bits_per_symbol=1)

def generate_sionna_batch(batch_size, ebno_db):
    """Generates a mixed batch of modulations for training."""
    x_batch = []
    y_batch = []
    
    samples_per_mod = batch_size // NUM_CLASSES
    if samples_per_mod == 0: samples_per_mod = 1
    
    source = BinarySource()
    channel = AWGN()

    for idx, mod in enumerate(MODULATIONS):
        const = get_sionna_constellation(mod)
        mapper = Mapper(constellation=const)
        
        b = source([samples_per_mod, INPUT_LENGTH * const.num_bits_per_symbol])
        x = mapper(b)
        y = channel([x, ebno_db])
        
        y_iq = tf.stack([tf.math.real(y), tf.math.imag(y)], axis=-1)
        x_batch.append(y_iq)
        y_batch.append(tf.fill([samples_per_mod], idx))
        
    return tf.concat(x_batch, axis=0), tf.concat(y_batch, axis=0)

def main():
    model = OpalVanguardModel(NUM_CLASSES)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    print("\n--- Phase 1: Training on Sionna Differentiable PHY (AWGN) ---")
    # Training Loop
    for epoch in range(EPOCHS):
        x_train, y_train = generate_sionna_batch(BATCH_SIZE * NUM_CLASSES, ebno_db=20.0)
        history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, verbose=1)
        print(f"Epoch {epoch+1}/{EPOCHS} complete.")

    # Save Model
    model.save('opal_vanguard_base.h5')
    print("\nModel saved as opal_vanguard_base.h5")

if __name__ == "__main__":
    main()
