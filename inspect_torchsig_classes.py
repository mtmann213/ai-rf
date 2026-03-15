import h5py
import numpy as np
from torchsig.signals.signal_lists import SIGNALS_SHARED_LIST

def inspect():
    classes = SIGNALS_SHARED_LIST
    print(f"Total TorchSig Classes: {len(classes)}")
    print("Full List:")
    for i, name in enumerate(classes):
        print(f"  {i:2d}: {name}")

    # RadioML 2018.01A Classes
    radioml_classes = [
        '32PSK', '16APSK', '32QAM', 'FM', 'GMSK', '32APSK', 'OQPSK', '8ASK',
        'BPSK', '8PSK', 'AM-SSB-SC', '4ASK', '16PSK', '64APSK', '128QAM',
        '128APSK', 'AM-DSB-SC', 'AM-SSB-WC', '64QAM', 'QPSK', '256QAM',
        'AM-DSB-WC', 'OOK', '16QAM'
    ]
    
    print("\nOverlap with RadioML 2018:")
    overlap = []
    missing = []
    for r_mod in radioml_classes:
        # Normalize name for comparison (torchsig uses lowercase usually)
        found = False
        for t_mod in classes:
            if r_mod.lower() == t_mod.lower() or r_mod.lower().replace('-', '') == t_mod.lower():
                overlap.append((r_mod, t_mod))
                found = True
                break
        if not found:
            missing.append(r_mod)
            
    print(f"  Overlap count: {len(overlap)}")
    for r, t in overlap:
        print(f"    {r} -> {t}")
        
    print(f"\n  Missing in TorchSig: {missing}")

if __name__ == "__main__":
    inspect()
