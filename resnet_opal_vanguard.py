import tensorflow as tf
from tensorflow.keras import layers, models

def residual_block(x, filters, kernel_size=3, stride=1):
    """
    A standard ResNet residual block with skip connections.
    """
    shortcut = x
    
    # First Convolution
    x = layers.Conv1D(filters, kernel_size, strides=stride, padding='same', 
                      kernel_initializer='he_normal',
                      kernel_constraint=tf.keras.constraints.MaxNorm(3))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Second Convolution
    x = layers.Conv1D(filters, kernel_size, strides=1, padding='same', 
                      kernel_initializer='he_normal',
                      kernel_constraint=tf.keras.constraints.MaxNorm(3))(x)
    x = layers.BatchNormalization()(x)
    
    # Adjust shortcut if dimensions changed
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, strides=stride, padding='same', 
                                 kernel_initializer='he_normal',
                                 kernel_constraint=tf.keras.constraints.MaxNorm(3))(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
        
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

def build_resnet_vanguard(input_shape, num_classes):
    """
    Deep Residual Network for Modulation Classification.
    """
    inputs = layers.Input(shape=input_shape)
    
    # NEW: Immediate stabilization layer to catch numerical spikes
    x = layers.BatchNormalization()(inputs)
    
    # Initial Convolution
    x = layers.Conv1D(64, 7, strides=2, padding='same', 
                      kernel_initializer='he_normal',
                      kernel_constraint=tf.keras.constraints.MaxNorm(3))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(3, strides=2, padding='same')(x)
    
    # ResNet Blocks
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    
    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 128)
    
    x = residual_block(x, 256, stride=2)
    x = residual_block(x, 256)
    
    # Final Classification Head
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(512, activation='relu', 
                     kernel_initializer='he_normal',
                     kernel_constraint=tf.keras.constraints.MaxNorm(3))(x)
    x = layers.Dropout(0.5)(x)
    
    # OUTPUT RAW LOGITS
    outputs = layers.Dense(num_classes, 
                           kernel_initializer='he_normal',
                           kernel_constraint=tf.keras.constraints.MaxNorm(3))(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="OpalVanguard_ResNet")
    return model

if __name__ == "__main__":
    # Test building the model
    INPUT_SHAPE = (1024, 2)
    NUM_CLASSES = 24
    model = build_resnet_vanguard(INPUT_SHAPE, NUM_CLASSES)
    model.summary()
    print("\nOpalVanguard ResNet Architecture built successfully.")
