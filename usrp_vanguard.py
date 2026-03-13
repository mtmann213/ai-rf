import numpy as np
import uhd
import time

class USRPVanguardManager:
    """
    Manages Project Opal Vanguard USRP Hardware.
    Supports TX/RX selection and closed-loop calibration.
    """
    def __init__(self, tx_serial=None, rx_serial=None, sample_rate=1e6, freq=2.4e9):
        self.sample_rate = sample_rate
        self.freq = freq
        
        # Initialize TX
        tx_args = f"serial={tx_serial}" if tx_serial else ""
        self.tx_usrp = uhd.usrp.MultiUSRP(tx_args)
        
        # Initialize RX
        rx_args = f"serial={rx_serial}" if rx_serial else ""
        self.rx_usrp = uhd.usrp.MultiUSRP(rx_args)
        
        self.setup_hardware()

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

    def calibrate_loop(self, duration=2.0):
        """
        Sends a CW (Continuous Wave) tone to calibrate path loss and phase.
        """
        print(f"Opal Vanguard: Calibrating TX ({self.tx_usrp.get_usrp_rx_info()['serial']}) "
              f"-> RX ({self.rx_usrp.get_usrp_rx_info()['serial']})")
        
        # 1. Generate Tone
        num_samples = int(self.sample_rate * duration)
        t = np.arange(num_samples) / self.sample_rate
        tone = 0.5 * np.exp(1j * 2 * np.pi * 100e3 * t).astype(np.complex64) # 100kHz offset tone
        
        # 2. Setup Streamers
        st_args = uhd.usrp.StreamArgs("fc32", "sc16")
        tx_streamer = self.tx_usrp.get_tx_stream(st_args)
        rx_streamer = self.rx_usrp.get_rx_stream(st_args)
        
        # 3. Transmit and Receive
        metadata = uhd.types.TXMetadata()
        rx_metadata = uhd.types.RXMetadata()
        
        rx_buffer = np.zeros(num_samples, dtype=np.complex64)
        
        print("Starting calibration loop...")
        tx_streamer.send(tone, metadata)
        rx_streamer.recv(rx_buffer, rx_metadata)
        
        # Calculate Power for Calibration
        p_in = np.mean(np.abs(tone)**2)
        p_out = np.mean(np.abs(rx_buffer)**2)
        path_loss_db = 10 * np.log10(p_in / p_out)
        
        print(f"Calibration Complete. Path Loss: {path_loss_db:.2f} dB")
        return path_loss_db

if __name__ == "__main__":
    # Example Hardware Trinity:
    # 3449AC1, 3457464, 3457480
    manager = USRPVanguardManager(tx_serial="3449AC1", rx_serial="3457464")
    manager.calibrate_loop()
