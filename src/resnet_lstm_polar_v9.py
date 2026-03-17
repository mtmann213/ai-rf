import os
import tensorflow as tf
from tensorflow.keras import layers, models

def build_resnet_lstm_polar_v9(iq_shape=(1024, 2), polar_shape=(1024, 2), num_classes=57):
    """
    V9.2 Multi-Modal CLDNN: The 'Sovereign Eye'.
    Ensembles IQ (Spatial/Temporal) with Polar (Geometric/Ring) features.
    """
    
    # --- INPUT 1: Raw I/Q ---
    iq_input = layers.Input(shape=iq_shape, name="iq_input")
    
    # --- INPUT 2: Amplitude & Phase ---
    polar_input = layers.Input(shape=polar_shape, name="polar_input")

    def residual_block(x, filters, kernel_size=3, stride=1, prefix=""):
        shortcut = x
        x = layers.Conv1D(filters, kernel_size, strides=stride, padding='same', kernel_initializer='he_normal', name=f"{prefix}_conv1")(x)
        x = layers.BatchNormalization(name=f"{prefix}_bn1")(x)
        x = layers.Activation('relu', name=f"{prefix}_relu1")(x)
        x = layers.Conv1D(filters, kernel_size, strides=1, padding='same', kernel_initializer='he_normal', name=f"{prefix}_conv2")(x)
        x = layers.BatchNormalization(name=f"{prefix}_bn2")(x)
        
        if stride != 1 or shortcut.shape[-1] != filters:
            shortcut = layers.Conv1D(filters, 1, strides=stride, padding='same', kernel_initializer='he_normal', name=f"{prefix}_short")(shortcut)
            shortcut = layers.BatchNormalization(name=f"{prefix}_short_bn")(shortcut)
            
        x = layers.Add(name=f"{prefix}_add")([x, shortcut])
        return layers.Activation('relu', name=f"{prefix}_out")(x)

    # --- STREAM A: IQ ResNet Branch ---
    a = layers.Conv1D(64, 7, strides=2, padding='same', kernel_initializer='he_normal')(iq_input)
    a = layers.BatchNormalization()(a)
    a = layers.Activation('relu')(a)
    a = residual_block(a, 64, prefix="iq_r1")
    a = residual_block(a, 128, stride=2, prefix="iq_r2")
    
    # --- STREAM B: Polar Branch (Focused on Amplitude Rings) ---
    b = layers.Conv1D(64, 7, strides=2, padding='same', kernel_initializer='he_normal')(polar_input)
    b = layers.BatchNormalization()(b)
    b = layers.Activation('relu')(b)
    b = residual_block(b, 64, prefix="pol_r1")
    b = residual_block(b, 128, stride=2, prefix="pol_r2")

    # --- ENSEMBLE MERGE ---
    combined = layers.Concatenate(name="modality_merge")([a, b])
    
    # --- TEMPORAL RHYTHM: Bi-LSTM ---
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=False), name="temporal_ear")(combined)
    
    # --- CLASSIFICATION HEAD ---
    x = layers.Dense(256, activation='relu', kernel_initializer='he_normal')(x)
    x = layers.Dropout(0.4)(x) # Increased dropout for multi-modal complexity
    x = layers.Dense(128, activation='relu', kernel_initializer='he_normal')(x)
    
    outputs = layers.Dense(num_classes, activation=None, name="logits_output")(x)

    model = models.Model(inputs=[iq_input, polar_input], outputs=outputs, name="Vanguard_V9_2_Sovereign")
    return model

if __name__ == "__main__":
    model = build_resnet_lstm_polar_v9()
    model.summary()
    print("\nV9.2 Sovereign Architecture Verified: Multi-Modal IQ + Polar Streams Integrated.")
