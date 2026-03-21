import os
import tensorflow as tf
from tensorflow.keras import layers, models

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention and Normalization
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def build_sovereign_transformer_v10(input_shape=(1024, 2), num_classes=57):
    """
    V10.0 Sovereign Transformer: The 'Global Eye'.
    Hybrid CNN-Transformer architecture optimized for high-density modulation ID.
    """
    inputs = layers.Input(shape=input_shape)

    # --- 1. LOCAL FEATURE EXTRACTOR (CNN Stem) ---
    # We use a few conv layers to extract local temporal patterns and reduce sequence length
    x = layers.Conv1D(64, 7, strides=2, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    
    # --- 2. POSITIONAL ENCODING ---
    # Since Transformers are permutation invariant, we add position information
    num_steps = x.shape[1]
    projection_dim = x.shape[2]
    positions = tf.range(start=0, limit=num_steps, delta=1)
    pos_encoding = layers.Embedding(input_dim=num_steps, output_dim=projection_dim)(positions)
    x = x + pos_encoding

    # --- 3. TRANSFORMER ENCODER BLOCKS ---
    # The 'Brain' of the model: Multi-Head Self-Attention
    for _ in range(4):
        x = transformer_encoder(x, head_size=128, num_heads=4, ff_dim=256, dropout=0.1)

    # --- 4. CLASSIFICATION HEAD ---
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation=None)(x) # Logits for stability

    model = models.Model(inputs, outputs, name="Sovereign_Transformer_V10")
    return model

if __name__ == "__main__":
    model = build_sovereign_transformer_v10()
    model.summary()
    print("\nV10.0 Sovereign Transformer Architecture Verified.")
