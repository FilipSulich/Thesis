import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import numpy as np
from data_cleaning.data_cleaner import DataCleaner


def preprocess_dataset(
    input_path: str,
    output_path: str,
    split: str = "train",
    check_quality: bool = True,
):
    cleaner = DataCleaner()

    with h5py.File(input_path, "r") as f_in:
        signals = f_in[f"{split}/signals"]
        labels = f_in[f"{split}/labels"]
        patient_ids = f_in[f"{split}/patient_ids"]
        segment_ids = f_in[f"{split}/segment_ids"]
        lead_times = f_in[f"{split}/lead_times"]

        n_total = len(signals)
        print(f"Processing {n_total} signals from {split} split...")

        clean_signals = []
        clean_labels = []
        clean_patient_ids = []
        clean_segment_ids = []
        clean_lead_times = []
        rejected_info = []

        for i in range(n_total):
            signal = signals[i][:]
            cleaned, info = cleaner.preprocess(signal, check_quality=check_quality)

            if cleaned is not None:
                clean_signals.append(cleaned)
                clean_labels.append(labels[i])
                clean_patient_ids.append(patient_ids[i])
                clean_segment_ids.append(segment_ids[i])
                clean_lead_times.append(lead_times[i])
            else:
                rejected_info.append({
                    "index": i,
                    "segment_id": segment_ids[i],
                    "reason": info.get("reason", "unknown"),
                })

            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{n_total}...")

    n_passed = len(clean_signals)
    n_rejected = len(rejected_info)
    print(f"\nDone: {n_passed} passed, {n_rejected} rejected ({100*n_rejected/n_total:.1f}%)")

    with h5py.File(output_path, "w") as f_out:
        grp = f_out.create_group(split)
        grp.create_dataset("signals", data=np.array(clean_signals))
        grp.create_dataset("labels", data=np.array(clean_labels))
        grp.create_dataset("patient_ids", data=np.array(clean_patient_ids))
        grp.create_dataset("segment_ids", data=np.array(clean_segment_ids))
        grp.create_dataset("lead_times", data=np.array(clean_lead_times))

    print(f"Saved cleaned dataset to {output_path}")
    return rejected_info


if __name__ == "__main__":
    input_path = "/Users/filipsulich/Developer/uni/Thesis/Thesis/data/extracted/unified_af_dataset.h5"
    output_path = "/Users/filipsulich/Developer/uni/Thesis/Thesis/data/extracted/unified_af_dataset_clean_2.h5"

    rejected = preprocess_dataset(input_path, output_path)