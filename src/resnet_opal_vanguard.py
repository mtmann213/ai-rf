import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def residual_block(x, filters, kernel_size=3, stride=1):
    shortcut = x
    l2_reg = regularizers.l2(1e-4)
    
    # First Convolution
    x = layers.Conv1D(filters, kernel_size, strides=stride, padding='same', 
                      kernel_initializer='he_uniform',
                      kernel_constraint=tf.keras.constraints.MaxNorm(3))(x)
    x = layers.LayerNormalization(axis=1, epsilon=0.1)(x) # Event Horizon: LN instead of BN
    x = layers.ReLU(max_value=6.0)(x) 
    
    # Second Convolution
    x = layers.Conv1D(filters, kernel_size, strides=1, padding='same', 
                      kernel_initializer='he_uniform',
                      kernel_constraint=tf.keras.constraints.MaxNorm(3))(x)
    x = layers.LayerNormalization(axis=1, epsilon=0.1)(x) # Event Horizon: LN instead of BN
    
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, strides=stride, padding='same', 
                                 kernel_initializer='he_uniform',
                                 kernel_constraint=tf.keras.constraints.MaxNorm(3))(shortcut)
        shortcut = layers.LayerNormalization(axis=1, epsilon=0.1)(shortcut)
        
    x = layers.Add()([x, shortcut])
    x = layers.ReLU(max_value=6.0)(x)
    return x
def build_resnet_vanguard(input_shape=(1024, 2), num_classes=24):
    inputs = layers.Input(shape=input_shape)
    l2_reg = regularizers.l2(1e-4)
    
    # Pre-Stabilizer: Individual Signal Normalization
    # Using epsilon=0.1 as a 'mathematical shield' for zero-variance signals.
    x = layers.LayerNormalization(axis=-1, epsilon=0.1)(inputs)
    x = layers.GaussianNoise(0.01)(x) # Force feature learning over mode collapse
    
    # Stem
    x = layers.Conv1D(64, 7, strides=2, padding='same', 
                      kernel_initializer='he_uniform',
                      kernel_constraint=tf.keras.constraints.MaxNorm(3))(x)
    x = layers.LayerNormalization(axis=1, epsilon=0.1)(x) # Event Horizon: LN instead of BN
    x = layers.ReLU(max_value=6.0)(x)
    x = layers.MaxPooling1D(3, strides=2, padding='same')(x)
    
    # Blocks
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 128)
    x = residual_block(x, 256, stride=2)
    x = residual_block(x, 256)
    
    # Head
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(512, kernel_initializer='he_uniform',
                     kernel_constraint=tf.keras.constraints.MaxNorm(3))(x)
    x = layers.ReLU(max_value=6.0)(x)
    x = layers.Dropout(0.5)(x)
    
    # Logits Output
    outputs = layers.Dense(num_classes, activation='softmax', kernel_initializer='glorot_uniform', name=f"dense_head_{num_classes}")(x)
    
    return models.Model(inputs=inputs, outputs=outputs, name=f"OpalVanguard_ResNet_V{num_classes}")
