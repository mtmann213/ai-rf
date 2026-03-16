import os
import sys
import tensorflow as tf
from data_loader import RadioMLDataLoader
from resnet_opal_vanguard import build_resnet_vanguard

# Enable strict numeric checking to instantly catch NaN/Inf
# tf.debugging.enable_check_numerics()

DATASET_PATH = "2018_01A/GOLD_XYZ_OSC.0001_1024.hdf5"
INPUT_SHAPE = (1024, 2)
NUM_CLASSES = 24
BATCH_SIZE = 64

def main():
    print("Starting fast NaN debugging run (max 800 steps)...")
    loader = RadioMLDataLoader(DATASET_PATH)
    train_indices, _ = loader.get_train_val_indices()
    
    # We use take(800) to ensure the dataset generator stops early
    train_ds = tf.data.Dataset.from_generator(
        lambda: loader.get_generator(train_indices, BATCH_SIZE),
        output_signature=(
            tf.TensorSpec(shape=(None, 1024, 2), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 24), dtype=tf.float32)
        )
    ).take(800).prefetch(tf.data.AUTOTUNE)

    model = build_resnet_vanguard(INPUT_SHAPE, NUM_CLASSES)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.00002, 
        global_clipnorm=1.0,
        clipvalue=0.1
    )
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    
    # JIT disabled for precision as per V7.0
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'], jit_compile=False)
    
    # Limit to 800 steps to reproduce the NaN seen at step 574
    model.fit(train_ds, steps_per_epoch=800, epochs=1)
    print("Debug run complete.")

if __name__ == "__main__":
    main()
