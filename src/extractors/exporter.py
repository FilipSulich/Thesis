import h5py
import numpy as np
import csv
from typing import List, Dict, Optional

class SegmentExporter:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def find_all_segments(
        self,
        split: str = "train",
        lead_times: Optional[List[int]] = None,
    ) -> Dict[str, List[Dict]]:
        """
        Extract all pre-AF segments (from all or specified lead times) and all normal segments.

        Args:
            split: Dataset split ("train", "val", or "test")
            lead_times: List of lead times to include (e.g., [5, 10, 15, 30, 45]).
                       If None, includes all pre-AF segments regardless of lead time.

        Returns:
            Dict with keys "pre_af" and "normal", each a list of segment dicts.
        """
        with h5py.File(self.file_path, "r") as f:
            lead_times_data = f[f"{split}/lead_times"][:]
            patient_ids = f[f"{split}/patient_ids"][:]
            segment_ids = f[f"{split}/segment_ids"][:]

            def decode(x):
                return x.decode() if isinstance(x, bytes) else str(x)

            pre_af_segments = []
            normal_segments = []

            for i in range(len(lead_times_data)):
                pid = decode(patient_ids[i])
                seg_id = decode(segment_ids[i])
                lt = int(lead_times_data[i])

                seg = {
                    "index": i,
                    "patient_id": pid,
                    "segment_id": seg_id,
                    "lead_time": lt,
                    "is_normal": lt == 0,
                }

                if lt == 0:
                    # Normal segment
                    normal_segments.append(seg)
                else:
                    # Pre-AF segment
                    if lead_times is None or lt in lead_times:
                        pre_af_segments.append(seg)

            print(f"Found {len(pre_af_segments)} pre-AF segments and {len(normal_segments)} normal segments")
            return {"pre_af": pre_af_segments, "normal": normal_segments}

    def get_signals(self, indices: List[int], split: str = "train") -> np.ndarray:
        with h5py.File(self.file_path, "r") as f:
            signals = f[f"{split}/signals"]
            return np.array([signals[i][:] for i in indices])

    def export_to_csv(
        self,
        output_path: str,
        split: str = "train",
        lead_times: Optional[List[int]] = None,
    ):
        """
        Export all pre-AF and normal segments to CSV for RQA analysis.

        Args:
            output_path: Path to output CSV file
            split: Dataset split ("train", "val", or "test")
            lead_times: List of lead times to include. If None, includes all pre-AF segments.
        """
        segments_dict = self.find_all_segments(split=split, lead_times=lead_times)

        all_segments = segments_dict["pre_af"] + segments_dict["normal"]
        indices = [s["index"] for s in all_segments]
        signals = self.get_signals(indices, split=split)

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["index", "patient_id", "segment_id", "lead_time", "is_normal"] + [f"s{i}" for i in range(signals.shape[1])]
            writer.writerow(header)

            for seg, sig in zip(all_segments, signals):
                writer.writerow([seg["index"], seg["patient_id"], seg["segment_id"], seg["lead_time"], seg["is_normal"]] + sig.tolist())

        print(f"Exported {len(segments_dict['pre_af'])} pre-AF + {len(segments_dict['normal'])} normal segments to {output_path}")

        # Print breakdown by lead time
        from collections import Counter
        lead_time_counts = Counter(s["lead_time"] for s in segments_dict["pre_af"])
        print(f"\nPre-AF segments by lead time:")
        for lt in sorted(lead_time_counts.keys()):
            print(f"  {lt} min: {lead_time_counts[lt]}")
        print(f"Normal segments: {len(segments_dict['normal'])}")


# Example usage
if __name__ == "__main__":
    # Use the unified_af_dataset_clean_2.h5 file
    path = "/Users/filipsulich/Developer/uni/Thesis/Thesis/data/extracted/unified_af_dataset_clean_2.h5"
    exporter = SegmentExporter(path)

    # Export all pre-AF segments (from all lead times) and all normal segments
    exporter.export_to_csv(
        output_path="/Users/filipsulich/Developer/uni/Thesis/Thesis/data/extracted/rqa_dataset_all_leadtimes.csv",
        split="train",
        lead_times=None,  # Include all lead times: 5, 10, 15, 30, 45 minutes
    )

    # Alternatively, you can specify specific lead times:
    # exporter.export_to_csv(
    #     output_path="/Users/filipsulich/Developer/uni/Thesis/Thesis/data/extracted/rqa_dataset_specific_leadtimes.csv",
    #     split="train",
    #     lead_times=[5, 15, 45],  # Only these lead times
    # )