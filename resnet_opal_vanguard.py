import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def residual_block(x, filters, kernel_size=3, stride=1):
    shortcut = x
    l2_reg = regularizers.l2(1e-4)
    
    # First Convolution
    x = layers.Conv1D(filters, kernel_size, strides=stride, padding='same', 
                      kernel_initializer='glorot_uniform',
                      kernel_constraint=tf.keras.constraints.MaxNorm(3))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Second Convolution
    x = layers.Conv1D(filters, kernel_size, strides=1, padding='same', 
                      kernel_initializer='glorot_uniform',
                      kernel_constraint=tf.keras.constraints.MaxNorm(3))(x)
    x = layers.BatchNormalization()(x)
    
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, strides=stride, padding='same', 
                                 kernel_initializer='glorot_uniform',
                                 kernel_constraint=tf.keras.constraints.MaxNorm(3))(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
        
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

def build_resnet_vanguard(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    
    # Pre-Stabilizer
    x = layers.LayerNormalization()(inputs)
    
    # Stem
    x = layers.Conv1D(64, 7, strides=2, padding='same', 
                      kernel_initializer='glorot_uniform',
                      kernel_constraint=tf.keras.constraints.MaxNorm(3))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
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
    x = layers.Dense(512, kernel_initializer='glorot_uniform',
                     kernel_constraint=tf.keras.constraints.MaxNorm(3))(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)
    
    # Logits Output (No Softmax)
    outputs = layers.Dense(num_classes, kernel_initializer='glorot_uniform')(x)
    
    return models.Model(inputs=inputs, outputs=outputs, name="OpalVanguard_ResNet_V6_Singularity")
