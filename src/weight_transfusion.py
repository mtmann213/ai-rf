import os
import tensorflow as tf
from src.resnet_lstm_v9 import build_resnet_lstm_v9
from src.resnet_lstm_polar_v9 import build_resnet_lstm_polar_v9

SOURCE_WEIGHTS = "models/v9_specialist_checkpoint.keras"
TARGET_MODEL_PATH = "models/v9_transfused_sovereign.keras"

def perform_transfusion():
    print(f"Opal Vanguard: Initiating SURGICAL Weight Transfusion")
    source_model = build_resnet_lstm_v9(num_classes=57)
    source_model.load_weights(SOURCE_WEIGHTS)
    target_model = build_resnet_lstm_polar_v9(num_classes=57)

    # SURGICAL MAPPING (Source Index -> Target Name)
    mapping = {
        # Initial Stem
        1: "conv1d_9", 
        2: "batch_normalization_9",
        # Residual Block 1 (IQ branch)
        5: "iq_r1_conv1",
        6: "iq_r1_bn1",
        8: "iq_r1_conv2",
        9: "iq_r1_bn2",
        # Residual Block 2 (IQ branch)
        12: "iq_r2_conv1",
        13: "iq_r2_bn1",
        15: "iq_r2_conv2",
        16: "iq_r2_short",
        17: "iq_r2_bn2",
        18: "iq_r2_short_bn",
        # Brain Layers
        30: "temporal_ear",
        31: "dense_3",
        33: "dense_4",
        34: "logits_output"
    }

    count = 0
    for src_idx, target_name in mapping.items():
        try:
            src_layer = source_model.layers[src_idx]
            target_layer = target_model.get_layer(target_name)
            target_layer.set_weights(src_layer.get_weights())
            print(f" Transfused: Source[{src_idx}] ({src_layer.name}) -> Target[{target_name}]")
            count += 1
        except Exception as e:
            print(f" FAILED: Source[{src_idx}] -> Target[{target_name}] | Error: {e}")

    target_model.save_weights(TARGET_MODEL_PATH)
    print(f"\nTransfusion Complete: {count} surgical strikes successful.")
    print(f"Warm-start weights saved to {TARGET_MODEL_PATH}")

if __name__ == "__main__":
    perform_transfusion()
