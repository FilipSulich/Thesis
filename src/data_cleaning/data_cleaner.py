from scipy.signal import butter, filtfilt, iirnotch

class DataCleaner:
    def __init__(self, signals, lowcut=0.5, highcut=125, order=4):
        self.signals = signals
        self.lowcut = lowcut
        self.highcut = highcut
        self.order = order


    def normalize(self, signal):
        return (signal - signal.mean()) / signal.std()

    def bandpass_filter(self, ecg, fs):
        nyquist = 0.5 * fs
        low = self.lowcut / nyquist
        high = self.highcut / nyquist

        b, a = butter(self.order, [low, high], btype='bandpass')
        ecg_filtered = filtfilt(b, a, ecg)

        return ecg_filtered

    def preprocess(self, signal):
        signal_demeaned = signal - signal.mean()
        signal_bandpass = self.bandpass_filter(signal_demeaned, 250)
        return self.normalize(signal_bandpass)

    def preprocess_group(self):
        clean_signals = []
        for signal in self.signals:
            clean_signals.append(self.preprocess(signal))
        return clean_signals