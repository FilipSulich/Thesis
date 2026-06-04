import sys
import json
from pathlib import Path

import numpy as np
import wfdb
from scipy.signal import resample

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.wavelet_selection_tests.urp_discrminability.wavelet_evaluator import WaveletEvaluator

DATA_DIR = PROJECT_ROOT / 'data_container' / 'raw_data' / 'paf'
PARAMS_JSON = PROJECT_ROOT / 'data_container' / 'raw_data' / 'tau_m_params.json'
output_dir = PROJECT_ROOT / 'results' / 'urp_discrminability'
output_dir.mkdir(parents=True, exist_ok=True)


# configuration parameters
FS_ORIG = 128
FS_TARGET = 500
SEGMENT_SAMPLES = 5000
N_SEGS_PER_REC = 5
SUBBAND_NAMES = [f'D{i}' for i in range(2, 9)]


def load_ecg_segments():
    """
    load and segment ECG data for wavelet testing.
    """

    # gets the available data from teh raw_data directory
    all_files = list(DATA_DIR.glob('*.hea'))
    all_records = [f.stem for f in all_files]

    # pre-AF records (label=1)
    preaf_recs = [r for r in all_records if r.startswith('p')]

    # NSR records (label=0)
    nsr_recs = [r for r in all_records if r.startswith('n')]

    def extract_segments(record_id, label):
        """rxtract segments from one record.
        Args:
            record_id (str): record ID
            label (int): label (0=NSR, 1=Pre-AF)
        """
        try:
            rec = wfdb.rdrecord(str(DATA_DIR / record_id))
            raw = rec.p_signal[:, 0].astype(np.float64)

            # resample to target fs
            n = int(len(raw) * FS_TARGET / FS_ORIG)
            sig = resample(raw, n).astype(np.float64)

            # get evenly-spaced segments
            max_start = len(sig) - SEGMENT_SAMPLES
            if max_start <= 0:
                return [], [], []

            starts = np.linspace(0, max_start, 15, dtype=int)
            segs, labs, rec_ids = [], [], []

            for s in starts[:N_SEGS_PER_REC]:
                seg = sig[s:s + SEGMENT_SAMPLES]
                if len(seg) == SEGMENT_SAMPLES:
                    segs.append(seg)
                    labs.append(label)
                    rec_ids.append(record_id)

            return segs, labs, rec_ids

        except Exception as e:
            return [], [], []

    signals, labels, record_ids = [], [], []

    for rid in preaf_recs:
        s, l, r = extract_segments(rid, 1)
        signals.extend(s)
        labels.extend(l)
        record_ids.extend(r)

    for rid in nsr_recs:
        s, l, r = extract_segments(rid, 0)
        signals.extend(s)
        labels.extend(l)
        record_ids.extend(r)

    return np.array(signals), np.array(labels), record_ids


def load_embedding_params():
    """load tau/m parameters for each subband."""
    with open(PARAMS_JSON) as f:
        raw_params = json.load(f)

    embedding_params = {}

    if 'subband_tau' in raw_params and 'subband_m' in raw_params:
        for k in SUBBAND_NAMES:
            if k in raw_params['subband_tau'] and k in raw_params['subband_m']:
                embedding_params[k] = {
                    'tau': raw_params['subband_tau'][k],
                    'm': raw_params['subband_m'][k]
                }
    else:
        embedding_params = raw_params

    return embedding_params


def main():
    """
    main function to conduct the URP discriminability test (train the models using 5-fold cross-validation - note that it takes a long time)
    """

    signals, labels, record_ids = load_ecg_segments()
    embedding_params = load_embedding_params()

    evaluator = WaveletEvaluator(
        signals=signals,
        labels=labels,
        record_ids=record_ids,
        embedding_params=embedding_params,
        n_cv_splits=5,
        results_dir=RESULTS_DIR,
    )
    print("running the wavelet selection test...")
    df = evaluator.run_all()
    evaluator.save_results()
    print("experiment finished")

    stats = evaluator.perform_statistical_tests(alpha=0.05)
    evaluator.save_statistical_tests(stats)

    best = df.iloc[0]

    print(f"\nbest wavelet: {best['wavelet']} (median AUC: {best['median_auc']:.4f})")


if __name__ == '__main__':
    main()
