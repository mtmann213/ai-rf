import os
import numpy as np
import uhd
import time
import tensorflow as tf

class USRPVanguardManager:
    """
    Manages Project Opal Vanguard USRP Hardware.
    Supports TX/RX selection, closed-loop calibration, and Neural Inference.
    """
    def __init__(self, tx_serial=None, rx_serial=None, sample_rate=1e6, freq=2.4e9, model_path='best_resnet_v7.keras'):
        self.sample_rate = sample_rate
        self.freq = freq
        self.modulations = [
            '32PSK', '16APSK', '32QAM', 'FM', 'GMSK', '32APSK', 'OQPSK', '8ASK',
            'BPSK', '8PSK', 'AM-SSB-SC', '4ASK', '16PSK', '64APSK', '128QAM',
            '128APSK', 'AM-DSB-SC', 'AM-SSB-WC', '64QAM', 'QPSK', '256QAM',
            'AM-DSB-WC', 'OOK', '16QAM'
        ]
        
        # 1. Initialize Hardware
        tx_args = f"serial={tx_serial}" if tx_serial else ""
        self.tx_usrp = uhd.usrp.MultiUSRP(tx_args)
        
        rx_args = f"serial={rx_serial}" if rx_serial else ""
        self.rx_usrp = uhd.usrp.MultiUSRP(rx_args)
        
        self.setup_hardware()

        # 2. Initialize Model
        self.model = None
        if os.path.exists(model_path):
            print(f"Opal Vanguard: Loading Neural Receiver from {model_path}...")
            self.model = tf.keras.models.load_model(model_path)
        else:
            print(f"Warning: Model {model_path} not found. AI Inference will be disabled.")

    def setup_hardware(self):
        """Sets common parameters for both TX and RX."""
        for usrp in [self.tx_usrp, self.rx_usrp]:
            usrp.set_rx_rate(self.sample_rate, 0)
            usrp.set_tx_rate(self.sample_rate, 0)
            usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(self.freq), 0)
            usrp.set_tx_freq(uhd.libpyuhd.types.tune_request(self.freq), 0)
            # Standard gains for closed-loop attenuated setup
            usrp.set_rx_gain(30, 0)
            usrp.set_tx_gain(30, 0)

    def normalize_soft_clip(self, x):
        """Event Horizon: Live Soft-clipping for inference stability."""
        # Convert complex to [I, Q]
        x_iq = np.stack([np.real(x), np.imag(x)], axis=-1)
        x_iq = np.nan_to_num(x_iq.astype(np.float32))
        return x_iq / (1.0 + np.abs(x_iq))

    def run_live_intercept(self, window_size=1024):
        """
        Infinite loop: Captures live signals from USRP and runs Neural Classification.
        """
        if self.model is None:
            print("Error: Cannot run intercept without a loaded model.")
            return

        print(f"\n--- Opal Vanguard: LIVE INTERCEPT ACTIVE ({self.freq/1e6:.1f} MHz) ---")
        
        # Setup Streamer
        st_args = uhd.usrp.StreamArgs("fc32", "sc16")
        rx_streamer = self.rx_usrp.get_rx_stream(st_args)
        
        # Buffers
        recv_buffer = np.zeros(window_size, dtype=np.complex64)
        metadata = uhd.types.RXMetadata()
        
        try:
            while True:
                # 1. Capture Signal
                num_rx = rx_streamer.recv(recv_buffer, metadata)
                if num_rx < window_size: continue
                
                # 2. Pre-process
                x_input = self.normalize_soft_clip(recv_buffer)
                x_input = np.expand_dims(x_input, axis=0) # Add batch dimension
                
                # 3. Neural Inference
                preds = self.model.predict(x_input, verbose=0)[0]
                class_idx = np.argmax(preds)
                confidence = preds[class_idx]
                
                # 4. Report
                if confidence > 0.5: # Threshold for reporting
                    print(f"[{time.strftime('%H:%M:%S')}] DETECTED: {self.modulations[class_idx]:<10} | Conf: {confidence:.2%}")
                
        except KeyboardInterrupt:
            print("\nIntercept terminated by user.")

    def calibrate_loop(self, duration=2.0):
        """Sends a CW tone to calibrate path loss."""
        print(f"Opal Vanguard: Calibrating TX -> RX")
        num_samples = int(self.sample_rate * duration)
        t = np.arange(num_samples) / self.sample_rate
        tone = 0.5 * np.exp(1j * 2 * np.pi * 100e3 * t).astype(np.complex64)
        
        st_args = uhd.usrp.StreamArgs("fc32", "sc16")
        tx_streamer = self.tx_usrp.get_tx_stream(st_args)
        rx_streamer = self.rx_usrp.get_rx_stream(st_args)
        
        metadata = uhd.types.TXMetadata()
        rx_metadata = uhd.types.RXMetadata()
        rx_buffer = np.zeros(num_samples, dtype=np.complex64)
        
        tx_streamer.send(tone, metadata)
        rx_streamer.recv(rx_buffer, rx_metadata)
        
        p_in = np.mean(np.abs(tone)**2)
        p_out = np.mean(np.abs(rx_buffer)**2)
        print(f"Path Loss: {10 * np.log10(p_in / p_out):.2f} dB")

if __name__ == "__main__":
    # Standard hardware configuration
    manager = USRPVanguardManager(
        tx_serial="3449AC1", 
        rx_serial="3457464", 
        freq=2.45e9, # Standard ISM band test
        sample_rate=1e6
    )
    
    # Run Live Intercept if model is ready
    if manager.model:
        manager.run_live_intercept()
    else:
        manager.calibrate_loop()
