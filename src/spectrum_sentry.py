import json
import time
import numpy as np
import tensorflow as tf
from usrp_vanguard import USRPVanguardManager

class SpectrumSentryBridge:
    """
    SIGINT Engine: Connects USRP Hardware to the ResNet Brain.
    Outputs real-time detections as a JSON stream.
    """
    def __init__(self, model_path='vanguard_final_production.keras', freq=2.45e9):
        # 1. Load the Brain
        print(f"Loading Neural Engine: {model_path}...")
        self.model = tf.keras.models.load_model(model_path)
        
        # 2. Setup the USRP "Ears"
        # Using rx_serial only for intercept mode
        self.manager = USRPVanguardManager(rx_serial="3457464", freq=freq)
        
        self.modulations = [
            '32PSK', '16APSK', '32QAM', 'FM', 'GMSK', '32APSK', 'OQPSK', '8ASK',
            'BPSK', '8PSK', 'AM-SSB-SC', '4ASK', '16PSK', '64APSK', '128QAM',
            '128APSK', 'AM-DSB-SC', 'AM-SSB-WC', '64QAM', 'QPSK', '256QAM',
            'AM-DSB-WC', 'OOK', '16QAM'
        ]

    def process_buffer(self, samples):
        """Converts raw complex samples to AI-ready soft-clipped IQ snapshots."""
        # Convert to [I, Q]
        iq = np.stack([np.real(samples), np.imag(samples)], axis=-1)
        # Soft-Clip Normalization (Matches training)
        iq = np.nan_to_num(iq.astype(np.float32))
        return iq / (1.0 + np.abs(iq))

    def run_intercept_stream(self):
        print("--- Spectrum Sentry: Streaming Detections ---")
        
        # USRP Streamer setup (matching hardware Trinity params)
        st_args = uhd.usrp.StreamArgs("fc32", "sc16")
        rx_streamer = self.manager.rx_usrp.get_rx_stream(st_args)
        
        recv_buffer = np.zeros(1024, dtype=np.complex64)
        metadata = uhd.types.RXMetadata()
        
        try:
            while True:
                # 1. Capture
                num_rx = rx_streamer.recv(recv_buffer, metadata)
                if num_rx < 1024: continue
                
                # 2. Infer
                input_data = self.process_buffer(recv_buffer)
                input_data = np.expand_dims(input_data, axis=0) # Add batch dim
                
                preds = self.model.predict(input_data, verbose=0)[0]
                class_idx = np.argmax(preds)
                confidence = float(preds[class_idx])
                
                # 3. Emit JSON
                detection = {
                    "timestamp": time.time(),
                    "frequency_mhz": self.manager.freq / 1e6,
                    "modulation": self.modulations[class_idx],
                    "confidence": confidence,
                    "power_db": float(10 * np.log10(np.mean(np.abs(recv_buffer)**2) + 1e-12))
                }
                
                # Print to stdout for external dashboard to pipe
                print(json.dumps(detection))
                
        except KeyboardInterrupt:
            print("\nSentry deactivated.")

if __name__ == "__main__":
    sentry = SpectrumSentryBridge(model_path='best_resnet_v7.keras')
    sentry.run_intercept_stream()
