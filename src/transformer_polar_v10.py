import os
import tensorflow as tf
from tensorflow.keras import layers, models

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0, prefix=""):
    # Attention and Normalization
    x = layers.LayerNormalization(epsilon=1e-6, name=f"{prefix}_ln1")(inputs)
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout, name=f"{prefix}_attn")(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6, name=f"{prefix}_ln2")(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu", name=f"{prefix}_ff1")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1, name=f"{prefix}_ff2")(x)
    return x + res

def build_sovereign_transformer_polar_v10(iq_shape=(1024, 2), polar_shape=(1024, 2), num_classes=57):
    """
    V10.1 Sovereign Polar Transformer: The 'Universal Eye'.
    Multi-modal attention ensembling IQ and Polar Geometric streams.
    """
    iq_input = layers.Input(shape=iq_shape, name="iq_input")
    polar_input = layers.Input(shape=polar_shape, name="polar_input")

    # --- STREAM A: IQ Attention ---
    a = layers.Conv1D(64, 7, strides=4, padding="same", activation="relu")(iq_input)
    a = layers.BatchNormalization()(a)
    # Positional Encoding
    pos_a = layers.Embedding(input_dim=a.shape[1], output_dim=a.shape[2])(tf.range(start=0, limit=a.shape[1], delta=1))
    a = a + pos_a
    for i in range(2):
        a = transformer_encoder(a, head_size=64, num_heads=4, ff_dim=128, dropout=0.1, prefix=f"iq_enc{i}")

    # --- STREAM B: Polar Attention ---
    b = layers.Conv1D(64, 7, strides=4, padding="same", activation="relu")(polar_input)
    b = layers.BatchNormalization()(b)
    # Positional Encoding
    pos_b = layers.Embedding(input_dim=b.shape[1], output_dim=b.shape[2])(tf.range(start=0, limit=b.shape[1], delta=1))
    b = b + pos_b
    for i in range(2):
        b = transformer_encoder(b, head_size=64, num_heads=4, ff_dim=128, dropout=0.1, prefix=f"pol_enc{i}")

    # --- ENSEMBLE FUSION ---
    # Global Average Pool both streams to get fixed-size vectors
    a_feat = layers.GlobalAveragePooling1D()(a)
    b_feat = layers.GlobalAveragePooling1D()(b)
    combined = layers.Concatenate(name="modality_merge")([a_feat, b_feat])
    
    # --- FINAL BRAIN ---
    x = layers.Dense(256, activation="relu", kernel_initializer='he_normal')(combined)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu", kernel_initializer='he_normal')(x)
    outputs = layers.Dense(num_classes, activation=None, name="logits_output")(x)

    model = models.Model(inputs=[iq_input, polar_input], outputs=outputs, name="Sovereign_Transformer_Polar_V10")
    return model

if __name__ == "__main__":
    model = build_sovereign_transformer_polar_v10()
    model.summary()
    print("\nV10.1 Multi-Modal Transformer Architecture Verified.")
