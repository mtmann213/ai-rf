import os
import tensorflow as tf
from tensorflow.keras import layers, models

def build_resnet_lstm_v9(input_shape=(1024, 2), num_classes=57):
    """
    V9.0 CLDNN Architecture: The 'Eyes + Ears' Ensemble.
    Combines 1D-ResNet (Spatial Features) with Bi-LSTM (Temporal Rhythm).
    """
    inputs = layers.Input(shape=input_shape)

    # --- THE EYES: Convolutional Feature Extractor (ResNet Style) ---
    def residual_block(x, filters, kernel_size=3, stride=1):
        shortcut = x
        x = layers.Conv1D(filters, kernel_size, strides=stride, padding='same', kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv1D(filters, kernel_size, strides=1, padding='same', kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        
        if stride != 1 or shortcut.shape[-1] != filters:
            shortcut = layers.Conv1D(filters, 1, strides=stride, padding='same', kernel_initializer='he_normal')(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
            
        x = layers.Add()([x, shortcut])
        return layers.Activation('relu')(x)

    # Initial Stem
    x = layers.Conv1D(64, 7, strides=2, padding='same', kernel_initializer='he_normal')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(3, strides=2, padding='same')(x)

    # Residual Stack (Eyes)
    x = residual_block(x, 64)
    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 256, stride=2)
    
    # --- THE EARS: Temporal Sequence Processor (LSTM) ---
    # We don't GlobalAveragePool yet; we want the sequence for the LSTM.
    # Shape here is (Length, Features) -> e.g., (64, 256)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=False))(x)
    
    # --- THE BRAIN: Deep Neural Network (DNN) ---
    x = layers.Dense(256, activation='relu', kernel_initializer='he_normal')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu', kernel_initializer='he_normal')(x)
    
    # Final Classification (Logits for Numerical Stability)
    outputs = layers.Dense(num_classes, activation=None)(x)

    model = models.Model(inputs, outputs, name="Vanguard_V9_CLDNN")
    return model

if __name__ == "__main__":
    # Test Build
    model = build_resnet_lstm_v9()
    model.summary()
    print("\nV9.0 CLDNN Architecture Verified: Eyes (CNN) + Ears (LSTM) Integrated.")
