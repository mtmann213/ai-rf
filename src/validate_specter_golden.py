import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from torchsig.utils.defaults import default_dataset
from torchsig.signals.signal_lists import SIGNALS_SHARED_LIST
from src.resnet_lstm_v9 import build_resnet_lstm_v9
from src.resnet_lstm_polar_v9 import build_resnet_lstm_polar_v9

# Configuration
MODELS_TO_TEST = [
    {"name": "V9.1_Specialist", "path": "models/vanguard_v9_specialist_final.keras", "type": "iq_only"},
    {"name": "V9.2_Sovereign", "path": "models/vanguard_v9_sovereign_final.keras", "type": "multi_modal"},
    {"name": "V9.4_Transfused", "path": "models/v9_specialist_sovereign_checkpoint.keras", "type": "multi_modal"}
]
NUM_CLASSES = 57
SAMPLES_TO_GEN = 2000 # Smaller sample for a fast pulse
IQ_LENGTH = 1024

def generate_synthetic_validation():
    print(f"Generating {SAMPLES_TO_GEN} synthetic snapshots (Level 1)...")
    
    x_val = np.zeros((SAMPLES_TO_GEN, IQ_LENGTH, 2), dtype=np.float32)
    y_val = np.zeros((SAMPLES_TO_GEN, NUM_CLASSES), dtype=np.float32)
    
    class_to_idx = {name: i for i, name in enumerate(SIGNALS_SHARED_LIST)}
    
    dataset = default_dataset(
        impairment_level=1,
        num_samples_dataset=SAMPLES_TO_GEN
    )
    it = iter(dataset)
    
    for i in tqdm(range(SAMPLES_TO_GEN), desc="Generating"):
        try:
            # item is a Signal object
            item = next(it)
            data = item.data
            
            # Extract class name from metadata
            try:
                class_name = item.component_signals[0]._metadata['class_name']
            except (AttributeError, IndexError):
                class_name = "unknown"
            
            if len(data) > IQ_LENGTH: data = data[:IQ_LENGTH]
            elif len(data) < IQ_LENGTH: data = np.pad(data, (0, IQ_LENGTH - len(data)), 'constant')
            
            iq = np.stack([np.real(data), np.imag(data)], axis=-1)
            iq = iq / (1.0 + np.abs(iq)) # Soft-Clip
            
            x_val[i] = iq
            if class_name in class_to_idx:
                y_val[i, class_to_idx[class_name]] = 1.0
        except StopIteration:
            break
            
    return x_val, y_val

def run_evaluation(model_info, x_raw, y_raw):
    print(f"\n--- Testing Synthetic Potential: {model_info['name']} ---")
    if model_info['type'] == "iq_only":
        model = build_resnet_lstm_v9(num_classes=NUM_CLASSES)
    else:
        model = build_resnet_lstm_polar_v9(num_classes=NUM_CLASSES)
    
    if os.path.exists(model_info['path']):
        model.load_weights(model_info['path'])
    else:
        print(f" ERROR: Weights not found at {model_info['path']}")
        return None

    if model_info['type'] == "multi_modal":
        i_comp, q_comp = x_raw[:, :, 0], x_raw[:, :, 1]
        amplitude = np.sqrt(i_comp**2 + q_comp**2)
        amplitude = amplitude / (1.0 + np.abs(amplitude))
        phase = np.arctan2(q_comp, i_comp) / np.pi
        x_polar = np.stack([amplitude, phase], axis=-1)
        inputs = [x_raw, x_polar]
    else:
        inputs = x_raw

    predictions = model.predict(inputs, batch_size=128)
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(y_raw, axis=1)
    
    # DEBUG: Show class name mapping for the first 5 samples
    print("\n--- Diagnostic Mapping (First 5) ---")
    for i in range(5):
        true_name = SIGNALS_SHARED_LIST[y_true[i]]
        # Need to handle if y_pred is outside the range of SIGNALS_SHARED_LIST (unlikely but safe)
        pred_name = SIGNALS_SHARED_LIST[y_pred[i]] if y_pred[i] < len(SIGNALS_SHARED_LIST) else f"Unknown_{y_pred[i]}"
        print(f" Sample {i}: True={true_name} | Pred={pred_name}")
    
    acc = np.mean(y_true == y_pred) * 100
    print(f" Synthetic (57-class) Accuracy: {acc:.2f}%")
    return acc

def main():
    x_val, y_val = generate_synthetic_validation()
    
    results = []
    for model_info in MODELS_TO_TEST:
        acc = run_evaluation(model_info, x_val, y_val)
        if acc:
            results.append((model_info['name'], acc))
            
    print("\n--- SYNTHETIC POTENTIAL LEADERBOARD ---")
    for name, acc in results:
        print(f" {name}: {acc:.2f}%")

if __name__ == "__main__":
    main()
