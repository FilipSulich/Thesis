import h5py
import pandas as pd


class DataReader:
    def __init__(self, file_path, file_path_csv, sampling_rate=250):
        self.file_path = file_path
        self.file_path_csv = file_path_csv
        self.sampling_rate = sampling_rate

    def read_data(self, index, split="train"):
        with h5py.File(self.file_path, "r") as f:
            signals = f[f"{split}/signals"]
            labels = f[f"{split}/labels"]
            patient_ids = f[f"{split}/patient_ids"]
            segment_ids = f[f"{split}/segment_ids"]
            lead_times = f[f"{split}/lead_times"]

            if index >= len(signals):
                print("Index out of range.")
                return None
            is_normal = False
            if lead_times[index] == -1:
                is_normal = True
            return {
                "raw_signal": [signals[index][:]],
                "label": [labels[index]],
                "patient_id": [patient_ids[index]],
                "segment_id": [segment_ids[index]],
                "lead_time": [lead_times[index]],
                "is_normal": [is_normal],
            }


    def read_data_csv(self, is_normal):
        data = pd.read_csv(self.file_path_csv)

        normal_signals = []
        preaf_signals = []
        # for index, row in data.iterrows():
            # print(index,row)
        return data


if __name__ == "__main__":
    path = "/Users/filipsulich/Developer/uni/Thesis/Thesis/data/extracted/matched_5min_50.csv"
    reader = DataReader(file_path=path, file_path_csv=path, sampling_rate=250)
    data = reader.read_data_csv(is_normal=True)
    signals = data.iloc[:,5:]
    print(signals)


