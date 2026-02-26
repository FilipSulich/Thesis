import h5py
import numpy as np
from typing import List, Dict, Optional


class DatasetInspector:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def find_segments(
        self,
        split: str = "train",
        is_normal: bool = True,
        lead_time: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[Dict]:
        """
        Find segments matching criteria.

        Args:
            split: Dataset split (train/val/test)
            is_normal: If True, find normal segments. If False, find pre-AF.
            lead_time: For pre-AF, filter by lead time in minutes. Ignored for normal.
            limit: Max number of segments to return.

        Returns:
            List of dicts with segment info and indices.
        """
        with h5py.File(self.file_path, "r") as f:
            lead_times = f[f"{split}/lead_times"][:]
            patient_ids = f[f"{split}/patient_ids"][:]
            segment_ids = f[f"{split}/segment_ids"][:]

            # lead_time=0 means normal, lead_time>0 means pre-AF
            if is_normal:
                mask = lead_times == 0
            else:
                if lead_time is not None:
                    mask = lead_times == lead_time
                else:
                    mask = lead_times > 0

            indices = np.where(mask)[0]

            if limit is not None:
                indices = indices[:limit]

            results = []
            for idx in indices:
                results.append({
                    "index": int(idx),
                    "patient_id": patient_ids[idx].decode() if isinstance(patient_ids[idx], bytes) else str(patient_ids[idx]),
                    "segment_id": segment_ids[idx].decode() if isinstance(segment_ids[idx], bytes) else str(segment_ids[idx]),
                    "lead_time": int(lead_times[idx]),
                    "is_normal": lead_times[idx] == 0,
                })

            return results

    def get_signals(self, indices: List[int], split: str = "train") -> np.ndarray:
        with h5py.File(self.file_path, "r") as f:
            signals = f[f"{split}/signals"]
            return np.array([signals[i][:] for i in indices])

    def export_matched_segments(
        self,
        output_path: str,
        split: str = "train",
        lead_time: int = 5,
        limit: int = 50,
        source_db: Optional[str] = None,
    ):
        """Export matched pre-AF and normal segments from same patients to CSV."""
        import csv

        matched = self.find_matched_pairs(split=split, lead_time=lead_time, source_db=source_db)

        pre_af = matched["pre_af"][:limit]
        normal = matched["normal"][:limit]

        all_segments = pre_af + normal
        indices = [s["index"] for s in all_segments]
        signals = self.get_signals(indices, split=split)

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["index", "patient_id", "segment_id", "lead_time", "is_normal"] + [f"s{i}" for i in range(signals.shape[1])]
            writer.writerow(header)

            for seg, signal in zip(all_segments, signals):
                row = [seg["index"], seg["patient_id"], seg["segment_id"], seg["lead_time"], seg["is_normal"]] + signal.tolist()
                writer.writerow(row)

        print(f"Exported {len(pre_af)} pre-AF + {len(normal)} normal segments to {output_path}")

    def find_matched_pairs(
        self,
        split: str = "train",
        lead_time: int = 5,
        source_db: Optional[str] = None,
    ) -> Dict:
        with h5py.File(self.file_path, "r") as f:
            lead_times = f[f"{split}/lead_times"][:]
            patient_ids = f[f"{split}/patient_ids"][:]
            segment_ids = f[f"{split}/segment_ids"][:]

            def decode(x):
                return x.decode() if isinstance(x, bytes) else str(x)

            def extract_db(segment_id: str) -> str:
                return segment_id.split("_")[0]

            all_segments = []
            for i in range(len(lead_times)):
                seg_id = decode(segment_ids[i])
                all_segments.append({
                    "index": i,
                    "patient_id": decode(patient_ids[i]),
                    "segment_id": seg_id,
                    "lead_time": int(lead_times[i]),
                    "is_normal": lead_times[i] == 0,
                    "source_db": extract_db(seg_id),
                })

            patients_with_preaf = {}
            patients_with_normal = {}

            for seg in all_segments:
                pid = seg["patient_id"]
                db = seg["source_db"]

                if seg["lead_time"] == lead_time:
                    key = (pid, db)
                    if key not in patients_with_preaf:
                        patients_with_preaf[key] = []
                    patients_with_preaf[key].append(seg)

                if seg["is_normal"]:
                    key = (pid, db)
                    if key not in patients_with_normal:
                        patients_with_normal[key] = []
                    patients_with_normal[key].append(seg)

            matched_keys = set(patients_with_preaf.keys()) & set(patients_with_normal.keys())

            if source_db:
                matched_keys = {k for k in matched_keys if k[1] == source_db}
            else:
                db_counts = {}
                for pid, db in matched_keys:
                    db_counts[db] = db_counts.get(db, 0) + 1
                if db_counts:
                    source_db = max(db_counts, key=db_counts.get)
                    matched_keys = {k for k in matched_keys if k[1] == source_db}

            pre_af_segments = []
            normal_segments = []

            for key in matched_keys:
                pre_af_segments.extend(patients_with_preaf[key])
                n_preaf = len(patients_with_preaf[key])
                normal_segments.extend(patients_with_normal[key][:n_preaf])

            return {
                "source_db": source_db,
                "lead_time": lead_time,
                "n_patients": len(matched_keys),
                "pre_af": pre_af_segments,
                "normal": normal_segments,
            }


if __name__ == "__main__":
    path = "/Users/filipsulich/Developer/uni/Thesis/Thesis/data/extracted/unified_af_dataset_clean.h5"
    inspector = DatasetInspector(path)

    inspector.export_matched_segments(
        output_path="/Users/filipsulich/Developer/uni/Thesis/Thesis/data/extracted/matched_5min_50_2.csv",
        split="train",
        lead_time=5,
        limit=100,
    )