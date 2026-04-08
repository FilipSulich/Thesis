"""
Hilbert envelope variance experiment for wavelet selection.

Research question: which wavelet function best preserves the temporal
amplitude dynamics of pre-AF flickering in D3 and D4 subbands?

Theoretical basis:
  - Lan et al. (2020, Chaos 30, 053137): pre-AF flickering produces
    transient amplitude non-stationarity in atrial-frequency subbands
  - Gabor (1946): instantaneous amplitude via analytic signal is valid
    for narrowband signals; wavelet subbands satisfy this condition
  - Percival & Walden (2000): longer filters increase temporal smearing,
    reducing preservation of fast amplitude transients

Design:
  - Apply each candidate wavelet to all ECG segments
  - Extract D3 (8-16 Hz) and D4 (4-8 Hz) subbands via SWT reconstruction
  - Compute Hilbert instantaneous amplitude envelope per subband
  - Compute envelope variance per segment per patient (median across segments)
  - Test pre-AF vs NSR group difference via Mann-Whitney U + Cohen's d
  - Select wavelet maximising Cohen's d in D3 and D4
  - Report leakage (DER from sine test) alongside discriminability

Negative control: D2 (16-32 Hz) — flickering theory predicts no effect here.
If D2 shows comparable discriminability to D3/D4, the result is confounded.

Statistical design:
  - Patient-level medians as statistical unit (no pseudoreplication)
  - Mann-Whitney U (non-parametric, not assuming normality of RQA features)
  - Cohen's d as effect size
  - Holm correction across subbands (D2 control, D3, D4) per wavelet
  - No wavelet-level multiple comparison correction — each wavelet is
    treated as a separate confirmatory test of the same hypothesis

n_pre_af = 25   # p* patients from PAF Prediction Challenge
n_nsr    = 25   # n* patients from PAF Prediction Challenge
n_segments_per_patient = 10
segment_length = 1280  # 10 s at 128 Hz

Candidate wavelets:
  db:   db2, db4, db6, db8, db10
  sym:  sym4, sym6, sym8
  coif: coif1, coif2, coif3, coif4
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import hilbert, welch
from scipy.stats import mannwhitneyu
import pywt
import os
import h5py
from pathlib import Path
import wfdb

# ------------------------------------------------------------------ #
# Configuration                                                         #
# ------------------------------------------------------------------ #

WAVELETS = ['db2', 'db4', 'db6', 'db8', 'db10',
            'sym4', 'sym6', 'sym8',
            'coif1', 'coif2', 'coif3', 'coif4']

LEVEL      = 4
FS         = 128
BAND_NAMES = ['D1', 'D2', 'D3', 'D4', 'A4']
TARGET_BANDS = ['D2', 'D3', 'D4']  # D2 = negative control

OUTPUT_DIR = 'hilbert_wavelet_selection'
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_DIR     = Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/data_new/exp2')
HDF5_PATH    = DATA_DIR / 'paf_with_quality.h5'
RAW_DATA_DIR = Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/data/paf')


# ------------------------------------------------------------------ #
# Data loading helpers                                                  #
# ------------------------------------------------------------------ #

def load_af_segments_from_raw(
    raw_data_dir: Path = RAW_DATA_DIR,
    n_segments_per_record: int = 10,
    segment_duration_sec: int = 10,
    sampling_freq: int = 128,
    lead_index: int = 0
) -> dict:
    """
    Load AF segments directly from raw p*c.dat files.

    Since AF segments (p*c records) are not included in the HDF5 file,
    this function reads them directly from the original WFDB files.

    Parameters
    ----------
    raw_data_dir : Path
        Directory containing the raw .dat and .hea files
    n_segments_per_record : int
        Number of segments to extract from each record
    segment_duration_sec : int
        Duration of each segment in seconds
    sampling_freq : int
        Sampling frequency (should be 128 Hz for PAF dataset)
    lead_index : int
        ECG lead index to extract (0 for first lead)

    Returns
    -------
    segments_af : {patient_id: list of 1D np.ndarray segments}
        AF segments from p*c records
    """
    segments_af = {}
    seg_samples = int(segment_duration_sec * sampling_freq)

    # Find all p*c.hea files
    p_c_files = sorted(raw_data_dir.glob('p*c.hea'))

    print(f"Loading AF segments from {len(p_c_files)} p*c records...")

    for hea_file in p_c_files:
        record_id = hea_file.stem  # e.g., 'p01c'

        try:
            # Read the record
            record = wfdb.rdrecord(str(raw_data_dir / record_id))

            if record.fs != sampling_freq:
                print(f"  Skipping {record_id}: unexpected fs={record.fs}")
                continue

            raw_signal = record.p_signal[:, lead_index].astype(np.float32)
            n_possible = len(raw_signal) // seg_samples

            if n_possible == 0:
                continue

            # Extract segments
            n_extract = min(n_segments_per_record, n_possible)
            starts = np.linspace(0, len(raw_signal) - seg_samples, n_extract, dtype=int)

            # Create patient_id (e.g., 'p01c' -> 'paf_p01c')
            patient_id = f"paf_{record_id}"

            if patient_id not in segments_af:
                segments_af[patient_id] = []

            for start in starts:
                segment = raw_signal[start: start + seg_samples]
                # Basic z-score normalization
                segment = (segment - np.mean(segment)) / (np.std(segment) + 1e-9)
                segments_af[patient_id].append(segment)

        except Exception as e:
            print(f"  Error loading {record_id}: {e}")
            continue

    print(f"Loaded {len(segments_af)} AF patients from p*c records")
    for pid in sorted(segments_af.keys())[:5]:
        print(f"  {pid}: {len(segments_af[pid])} segments")

    return segments_af


def load_segments_from_hdf5(hdf5_path: Path) -> tuple[dict, dict, dict]:
    """
    Extract pre-AF, NSR, and AF segments from HDF5 file.

    Pre-AF segments: records with p* IDs where the number is even (p02, p04, ...)
    NSR segments: records with n* IDs from n01 to n25 only
    AF segments: records with p* IDs ending with 'c' (p01c, p02c, ...)

    Note: p-odd records (p01, p03, ...) are NSR from PAF patients, we exclude them.

    Returns
    -------
    segments_pre_af : {patient_id: list of 1D np.ndarray segments}
        25 patients with even p* numbers
    segments_nsr : {patient_id: list of 1D np.ndarray segments}
        25 patients with n* IDs (healthy controls, n01-n25)
    segments_af : {patient_id: list of 1D np.ndarray segments}
        Patients during AF (p*c records)
    """
    segments_pre_af = {}
    segments_nsr = {}
    segments_af = {}

    with h5py.File(hdf5_path, 'r') as f:
        grp = f['all']
        signals = grp['signals'][:]
        labels = grp['labels'][:]
        segment_ids = [s.decode() if isinstance(s, bytes) else s
                      for s in grp['segment_ids'][:]]
        patient_ids = (
            [s.decode() if isinstance(s, bytes) else s
             for s in grp['patient_ids'][:]]
            if 'patient_ids' in grp else [''] * len(signals)
        )
        record_ids = (
            [s.decode() if isinstance(s, bytes) else s
             for s in grp['record_ids'][:]]
            if 'record_ids' in grp else [''] * len(signals)
        )

        # Group segments by patient using record_id to distinguish p-even vs p-odd
        for i, (sig, label, seg_id, pat_id, rec_id) in enumerate(
            zip(signals, labels, segment_ids, patient_ids, record_ids)
        ):
            rec_id_lower = rec_id.lower()

            # Extract NSR segments (n01-n25 records only)
            if rec_id_lower.startswith('n') and not rec_id_lower.endswith('c'):
                try:
                    # Extract number from record_id (e.g., 'n01' -> 1)
                    num_str = ''
                    for char in rec_id_lower[1:]:  # Skip the 'n'
                        if char.isdigit():
                            num_str += char
                        else:
                            break

                    if num_str:
                        record_num = int(num_str)
                        # Only include n01-n25 (first 25 healthy controls)
                        if 1 <= record_num <= 25:
                            if pat_id not in segments_nsr:
                                segments_nsr[pat_id] = []
                            segments_nsr[pat_id].append(sig)
                except (ValueError, IndexError):
                    pass

            # Extract AF segments (p*c records - patients during AF)
            elif rec_id_lower.endswith('c') and rec_id_lower.startswith('p'):
                if pat_id not in segments_af:
                    segments_af[pat_id] = []
                segments_af[pat_id].append(sig)

            # Extract pre-AF segments (p-even records only, not ending with 'c')
            elif rec_id_lower.startswith('p') and not rec_id_lower.endswith('c'):
                try:
                    # Extract number from record_id (e.g., 'p02' -> 2)
                    num_str = ''
                    for char in rec_id_lower[1:]:  # Skip the 'p'
                        if char.isdigit():
                            num_str += char
                        else:
                            break

                    if num_str:
                        record_num = int(num_str)
                        # Only include even-numbered records (pre-AF)
                        if record_num % 2 == 0:
                            if pat_id not in segments_pre_af:
                                segments_pre_af[pat_id] = []
                            segments_pre_af[pat_id].append(sig)
                except (ValueError, IndexError):
                    # Skip if we can't parse the record number
                    pass

    print(f"Loaded {len(segments_pre_af)} pre-AF patients (p-even records)")
    print(f"Loaded {len(segments_nsr)} NSR patients (n01-n25 records)")
    print(f"Loaded {len(segments_af)} AF patients (p*c records)")

    return segments_pre_af, segments_nsr, segments_af


# ------------------------------------------------------------------ #
# SWT reconstruction helper                                             #
# ------------------------------------------------------------------ #

def swt_reconstruct_subbands(signal: np.ndarray,wavelet: str,level: int) -> dict:
    """
    Decompose signal into subbands via SWT reconstruction.
    Returns dict {band_name: subband_signal}.
    """
    coeffs = pywt.swt(signal, wavelet, level=level)
    # coeffs[0] = (cA_level, cD_level), coeffs[-1] = (cA_1, cD_1)
    # level 1 = highest frequency (D1), level N = lowest detail (D4)
    subbands = {}
    n = len(signal)

    for i in range(level):
        band = f'D{i+1}'
        target = [(np.zeros_like(cA), np.zeros_like(cD))
                  for cA, cD in coeffs]
        # coeffs ordered from coarsest to finest in pywt
        target[level - 1 - i] = (
            np.zeros_like(coeffs[level - 1 - i][0]),
            coeffs[level - 1 - i][1]
        )
        subbands[band] = pywt.iswt(target, wavelet)[:n]

    # A4: reconstruct from approximation coefficients only
    target = [(cA, np.zeros_like(cD)) for cA, cD in coeffs]
    # zero out all but coarsest approx
    for j in range(1, level):
        target[j] = (np.zeros_like(target[j][0]),
                     np.zeros_like(target[j][1]))
    subbands['A4'] = pywt.iswt(target, wavelet)[:n]

    return subbands


# ------------------------------------------------------------------ #
# Hilbert envelope variance                                             #
# ------------------------------------------------------------------ #

def hilbert_envelope_variance(signal: np.ndarray) -> float:
    """
    Compute variance of the instantaneous amplitude envelope.

    The analytic signal z(t) = signal + j*H{signal} has instantaneous
    amplitude |z(t)|. Its variance measures amplitude non-stationarity:
    high variance = signal amplitude fluctuates strongly over time,
    consistent with transient attractor-switching episodes.

    Valid only for narrowband signals — wavelet subbands satisfy this
    condition by construction (Gabor 1946; Boashash 1992).
    """
    analytic  = hilbert(signal)
    envelope  = np.abs(analytic)
    return float(np.var(envelope))


# ------------------------------------------------------------------ #
# Leakage metric from sine test (DER)                                   #
# ------------------------------------------------------------------ #

def compute_der(wavelet: str, level: int, fs: int,
                n: int = 1280) -> dict:
    """
    Diagonal Energy Ratio per subband.
    Inject isolated sine at each subband centre frequency,
    decompose, measure fraction of energy in the target band.

    DER = 1 means all energy correctly routed to target band.
    DER < 1 means energy leaked to other bands.
    """
    band_freqs = {'D1': 48, 'D2': 24, 'D3': 12, 'D4': 6, 'A4': 2}
    t   = np.arange(n) / fs
    der = {}

    for target_band, f0 in band_freqs.items():
        sine    = np.sin(2 * np.pi * f0 * t)
        subs    = swt_reconstruct_subbands(sine, wavelet, level)
        energies = {b: np.sum(subs[b]**2) for b in BAND_NAMES}
        total   = sum(energies.values()) + 1e-12
        der[target_band] = energies[target_band] / total

    return der


# ------------------------------------------------------------------ #
# Main experiment loop                                                  #
# ------------------------------------------------------------------ #

def run_hilbert_experiment(segments_pre_af: dict,
                           segments_nsr:    dict) -> pd.DataFrame:
    """
    Parameters
    ----------
    segments_pre_af : {patient_id: list of 1D np.ndarray segments}
    segments_nsr    : {patient_id: list of 1D np.ndarray segments}

    Returns
    -------
    DataFrame with one row per (wavelet, band) containing:
      cohen_d, u_stat, p_value, p_holm, significant_holm,
      mean_pre_af, mean_nsr, der
    """
    rows = []

    for wavelet in WAVELETS:
        print(f"\nProcessing wavelet: {wavelet}")

        # DER from sine test for this wavelet
        der = compute_der(wavelet, LEVEL, FS)

        # Patient-level envelope variance per band
        patient_env_var = {band: {'pre_af': [], 'nsr': []}
                           for band in TARGET_BANDS}

        for group_label, segments_dict in [
                ('pre_af', segments_pre_af),
                ('nsr',    segments_nsr)]:

            for pid, segs in segments_dict.items():
                band_vars = {band: [] for band in TARGET_BANDS}

                for seg in segs:
                    # Z-score normalise raw segment
                    seg_n = (seg - np.mean(seg)) / (np.std(seg) + 1e-9)
                    subs  = swt_reconstruct_subbands(
                        seg_n, wavelet, LEVEL)

                    for band in TARGET_BANDS:
                        sub = subs[band]
                        # Z-score subband before Hilbert
                        sub_n = (sub - np.mean(sub)) / \
                                (np.std(sub) + 1e-9)
                        ev = hilbert_envelope_variance(sub_n)
                        band_vars[band].append(ev)

                # Patient-level median across segments
                for band in TARGET_BANDS:
                    patient_env_var[band][group_label].append(
                        np.median(band_vars[band]))

        # Statistical tests per band
        band_results = []
        p_values     = []

        for band in TARGET_BANDS:
            x = np.array(patient_env_var[band]['pre_af'])
            y = np.array(patient_env_var[band]['nsr'])

            u, p = mannwhitneyu(x, y, alternative='two-sided')

            n1, n2 = len(x), len(y)
            pooled = np.sqrt(
                ((n1-1)*np.var(x, ddof=1) + (n2-1)*np.var(y, ddof=1))
                / (n1 + n2 - 2)
            )
            d = (np.mean(x) - np.mean(y)) / pooled \
                if pooled > 1e-9 else np.nan

            band_results.append({
                'wavelet':    wavelet,
                'band':       band,
                'cohen_d':    d,
                'u_stat':     u,
                'p_value':    p,
                'mean_pre_af': np.mean(x),
                'mean_nsr':    np.mean(y),
                'der':        der[band],
                'filter_len': len(pywt.Wavelet(wavelet).dec_lo),
                '_x': x,
                '_y': y,
            })
            p_values.append(p)

        # Holm correction across bands per wavelet
        n_tests = len(p_values)
        order   = np.argsort(p_values)
        alphas  = np.zeros(n_tests)
        for rank, idx in enumerate(order):
            alphas[idx] = 0.05 / (n_tests - rank)

        for i, res in enumerate(band_results):
            res['alpha_holm']       = alphas[i]
            res['significant_holm'] = res['p_value'] < alphas[i]

        rows.extend(band_results)

    return pd.DataFrame(rows)


# ------------------------------------------------------------------ #
# Results and plots                                                     #
# ------------------------------------------------------------------ #

def save_results(df: pd.DataFrame):
    # Drop array columns before saving
    stat_cols = [c for c in df.columns if not c.startswith('_')]
    path = os.path.join(OUTPUT_DIR, 'hilbert_results.csv')
    df[stat_cols].to_csv(path, index=False)
    print(f"Results saved to {path}")


def plot_cohens_d_by_wavelet(df: pd.DataFrame):
    """
    Line plot: Cohen's d vs wavelet (grouped by family),
    one line per band (D2=control, D3, D4).

    Expected pattern:
      - D3 and D4: peak at intermediate order, decline at high order
      - D2 (control): flat and near zero across all wavelets

    If D2 shows comparable Cohen's d to D3/D4, the result is confounded
    by broadband effects (HRV, QRS) rather than specific flickering dynamics.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    families  = {'db': 'Daubechies', 'sym': 'Symlet', 'coif': 'Coiflet'}
    colors    = {'D2': 'grey', 'D3': 'steelblue', 'D4': 'tomato'}
    linestyles = {'D2': '--', 'D3': '-', 'D4': '-'}

    for ax, (prefix, fname) in zip(axes, families.items()):
        fam_df = df[df['wavelet'].str.startswith(prefix)].copy()

        # Extract numeric order for x-axis
        fam_df['order'] = fam_df['wavelet'].str.replace(
            prefix, '', regex=False).astype(int)

        for band in TARGET_BANDS:
            band_df = fam_df[fam_df['band'] == band].sort_values('order')
            ax.plot(band_df['order'], band_df['cohen_d'],
                    marker='o', color=colors[band],
                    linestyle=linestyles[band],
                    linewidth=2, markersize=7,
                    label=band)
            # Mark significant points
            sig = band_df[band_df['significant_holm']]
            ax.scatter(sig['order'], sig['cohen_d'],
                       s=120, color=colors[band],
                       edgecolors='black', zorder=5,
                       marker='*')

        ax.axhline(0, color='black', linewidth=0.8, linestyle=':')
        ax.set_title(fname, fontweight='bold', fontsize=12)
        ax.set_xlabel("Wavelet order", fontsize=10)
        if ax == axes[0]:
            ax.set_ylabel("Cohen's d  (pre-AF − NSR)", fontsize=10)
        ax.legend(title='Subband', fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle(
        "Hilbert envelope variance discriminability: pre-AF vs NSR\n"
        "★ = significant after Holm correction  |  "
        "D2 is negative control subband",
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'cohens_d_by_wavelet.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_tradeoff(df: pd.DataFrame):
    """
    Scatter plot: DER (x-axis) vs Cohen's d D3 (y-axis).
    One point per wavelet, marker shape = family, colour = order.

    This plot shows the leakage-discriminability tradeoff:
    high DER = low leakage = good spectral isolation
    high Cohen's d = good temporal preservation of flickering

    The Pareto-optimal wavelet sits closest to upper-right corner.
    """
    d3_df = df[df['band'] == 'D3'].copy()

    families  = {'db': ('o', 'Daubechies'),
                 'sym': ('s', 'Symlet'),
                 'coif': ('^', 'Coiflet')}
    cmap = plt.cm.viridis

    fig, ax = plt.subplots(figsize=(9, 6))

    for prefix, (marker, fname) in families.items():
        fam = d3_df[d3_df['wavelet'].str.startswith(prefix)].copy()
        fam['order'] = fam['wavelet'].str.replace(
            prefix, '', regex=False).astype(int)
        orders_norm = (fam['order'] - fam['order'].min()) / \
                      (fam['order'].max() - fam['order'].min() + 1e-9)

        sc = ax.scatter(fam['der'], fam['cohen_d'],
                        c=orders_norm, cmap=cmap,
                        vmin=0, vmax=1,
                        s=120, marker=marker,
                        edgecolors='black', linewidths=0.8,
                        label=fname)

        for _, row in fam.iterrows():
            ax.annotate(row['wavelet'],
                        (row['der'], row['cohen_d']),
                        textcoords='offset points',
                        xytext=(5, 5), fontsize=8)

    # Pareto frontier
    d3_sorted = d3_df.sort_values('der')
    pareto_x, pareto_y = [], []
    best_d = -np.inf
    for _, row in d3_sorted.iterrows():
        if row['cohen_d'] > best_d:
            best_d = row['cohen_d']
            pareto_x.append(row['der'])
            pareto_y.append(row['cohen_d'])
    if len(pareto_x) > 1:
        ax.step(pareto_x, pareto_y, where='post',
                color='red', linewidth=1.5,
                linestyle='--', label='Pareto frontier',
                alpha=0.7)

    ax.set_xlabel("Diagonal Energy Ratio — D3  (higher = less leakage)",
                  fontsize=11)
    ax.set_ylabel("Cohen's d — envelope variance D3  (pre-AF vs NSR)",
                  fontsize=11)
    ax.set_title(
        "Leakage vs. discriminability tradeoff by wavelet\n"
        "Optimal wavelet = upper-right corner",
        fontsize=12, fontweight='bold'
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    cb = plt.colorbar(
        plt.cm.ScalarMappable(cmap=cmap,
                              norm=plt.Normalize(0, 1)),
        ax=ax)
    cb.set_label('Normalised order (0=low, 1=high)', fontsize=9)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'tradeoff_plot.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_envelope_distributions(df: pd.DataFrame,
                                best_wavelet: str):
    """
    For the selected wavelet: violin plots of envelope variance
    distributions across patients for pre-AF and NSR in D3 and D4.

    Shows the actual distributional difference that Cohen's d summarises.
    """
    # Re-extract raw arrays for best wavelet
    best = df[df['wavelet'] == best_wavelet]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    colors = {'pre_af': 'tomato', 'nsr': 'steelblue'}

    for ax, band in zip(axes, ['D3', 'D4']):
        row = best[best['band'] == band].iloc[0]
        data  = [row['_x'], row['_y']]
        parts = ax.violinplot(data, positions=[1, 2],
                              showmedians=True, showextrema=True)

        for body, color in zip(parts['bodies'],
                               [colors['pre_af'], colors['nsr']]):
            body.set_facecolor(color)
            body.set_alpha(0.7)
        for key in ('cmedians', 'cmins', 'cmaxes', 'cbars'):
            if key in parts:
                parts[key].set_color('black')
                parts[key].set_linewidth(1.2)

        # Significance annotation
        y_max = max(row['_x'].max(), row['_y'].max())
        sig   = '***' if row['p_value'] < 0.001 \
                else '**' if row['p_value'] < 0.01 \
                else '*'  if row['p_value'] < 0.05 \
                else 'ns'
        ax.plot([1, 1, 2, 2],
                [y_max*1.02, y_max*1.05, y_max*1.05, y_max*1.02],
                lw=1.2, color='black')
        ax.text(1.5, y_max*1.06, sig,
                ha='center', va='bottom', fontsize=12)

        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Pre-AF', 'NSR'], fontsize=11)
        ax.set_title(
            f"{band} subband — {best_wavelet}\n"
            f"Cohen's d = {row['cohen_d']:.3f}, "
            f"p = {row['p_value']:.4f}",
            fontweight='bold'
        )
        ax.set_ylabel("Hilbert envelope variance", fontsize=10)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle(
        f"Envelope variance distributions — selected wavelet: {best_wavelet}\n"
        f"n = {len(row['_x'])} pre-AF, {len(row['_y'])} NSR patients",
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR,
                        f'distributions_{best_wavelet}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def print_summary_table(df: pd.DataFrame):
    stat_cols = [c for c in df.columns if not c.startswith('_')]
    summary   = df[stat_cols][df['band'].isin(['D3', 'D4'])]
    print("\n" + "="*80)
    print("HILBERT ENVELOPE VARIANCE — SUMMARY TABLE (D3 and D4 only)")
    print("="*80)
    print(summary[[
        'wavelet', 'band', 'cohen_d', 'p_value',
        'significant_holm', 'mean_pre_af', 'mean_nsr',
        'der', 'filter_len'
    ]].to_string(index=False))


# ------------------------------------------------------------------ #
# Entry point                                                           #
# ------------------------------------------------------------------ #

if __name__ == '__main__':
    # Load segments from HDF5 file
    # segments_pre_af = {pid: [seg1, seg2, ...], ...}  25 patients (p-even)
    # segments_nsr    = {pid: [seg1, seg2, ...], ...}  25 patients (n01-n25)
    # segments_af     = {pid: [seg1, seg2, ...], ...}  0 patients (not in HDF5)
    segments_pre_af, segments_nsr, segments_af = load_segments_from_hdf5(HDF5_PATH)

    # Load AF segments from raw data (p*c records) if needed
    # Note: AF segments are not in the HDF5 file, use this function to load them:
    if len(segments_af) == 0:
        segments_af = load_af_segments_from_raw(n_segments_per_record=10)

    df = run_hilbert_experiment(segments_af, segments_nsr)
    save_results(df)
    print_summary_table(df)

    # Select best wavelet: max Cohen's d in D3 + D4 combined
    d34 = df[df['band'].isin(['D3', 'D4'])].groupby(
        'wavelet')['cohen_d'].mean()
    best_wavelet = d34.idxmax()
    print(f"\nSelected wavelet: {best_wavelet}")

    plot_cohens_d_by_wavelet(df)
    plot_tradeoff(df)
    plot_envelope_distributions(df, best_wavelet)