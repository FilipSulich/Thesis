"""
Scientific justification for SWT denoising level and decomposition level selection.

Tests:
  1. Coefficient normality test per level → denoising cutoff
     (Donoho & Johnstone 1994: threshold only where coefficients are Gaussian)

  2. Signal energy per level → where is the ECG energy concentrated

  3. Shannon entropy per level → which subbands contain structured dynamics

  4. SNR improvement per level → how much denoising helps vs hurts

  5. Oscillation cycle count per level → minimum cycles for reliable RP

  6. Wavelet variance explained per level → Parseval-based energy decomposition

Run on a random subset of Pre-AF and NSR records.
Outputs a figure and a printed table for your thesis methods section.
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pywt
import wfdb
from scipy import signal as sp_signal
from scipy import stats
from tqdm import tqdm

warnings.filterwarnings('ignore', category=FutureWarning, module='pywt')

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_DIR        = Path('/data/paf')
FS_ORIG         = 128
FS_TARGET       = 500
SEGMENT_SAMPLES = 5000
WAVELET         = 'db4'
MAX_LEVEL       = 10
PAD_SAMPLES     = 5120   # for level-10 SWT

OUT_DIR = Path('/results/exp3/level_selection')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Records to analyse
PREAF_RECORDS = ['p01', 'p03', 'p05', 'p07', 'p09', 'p11', 'p13', 'p15',
                 'p17', 'p19', 'p21', 'p23', 'p25', 'p27', 'p29']
NSR_RECORDS   = ['n01', 'n02', 'n03', 'n04', 'n05', 'n06', 'n07', 'n08',
                 'n09', 'n10', 'n11', 'n12', 'n13', 'n14', 'n15']


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_middle_segment(record_id):
    """Load middle 10-second segment from a record."""
    rec = wfdb.rdrecord(str(DATA_DIR / record_id))
    raw = rec.p_signal[:, 0].astype(np.float64)
    n_tgt = int(len(raw) * FS_TARGET / FS_ORIG)
    res   = sp_signal.resample(raw, n_tgt)
    mid   = max(0, (len(res) - SEGMENT_SAMPLES) // 2)
    seg   = res[mid:mid + SEGMENT_SAMPLES].astype(np.float32)
    # Robust min-max normalisation
    p1, p99 = np.percentile(seg, 1), np.percentile(seg, 99)
    seg = np.clip(seg, p1, p99)
    seg = (seg - seg.min()) / (seg.max() - seg.min() + 1e-9)
    # Pad for SWT
    pad = PAD_SAMPLES - len(seg)
    return np.pad(seg, (0, pad), mode='reflect')


def compute_level_stats(record_id, wavelet=WAVELET, max_level=MAX_LEVEL):
    """
    Compute per-level statistics for one record.
    Returns dict with arrays of length max_level.
    """
    sig = load_middle_segment(record_id)
    coeffs = pywt.swt(sig, wavelet, level=max_level)
    # coeffs[0] = deepest level (D10), coeffs[-1] = shallowest (D1)
    # Reverse so index 0 = D1, index 9 = D10
    coeffs_ordered = list(reversed(coeffs))

    n_levels = len(coeffs_ordered)
    results = {
        'normality_p':    np.zeros(n_levels),   # p-value of normality test
        'energy_frac':    np.zeros(n_levels),   # fraction of total signal energy
        'entropy':        np.zeros(n_levels),   # Shannon entropy of detail coefficients
        'coeff_std':      np.zeros(n_levels),   # std of detail coefficients
        'zero_crossing':  np.zeros(n_levels),   # zero crossing rate (proxy for frequency content)
        'snr_estimate':   np.zeros(n_levels),   # signal-to-noise estimate
    }

    total_energy = np.sum(sig ** 2)

    for level_idx in range(n_levels):
        cA, cD = coeffs_ordered[level_idx]
        d = cD[:SEGMENT_SAMPLES]  # trim to original length

        # 1. Normality test (D'Agostino K²)
        # p < 0.05 means NOT normal → signal content present
        # p > 0.05 means normal → looks like noise → safe to threshold
        if len(d) > 8:
            try:
                _, p = stats.normaltest(d)
                results['normality_p'][level_idx] = p
            except Exception:
                results['normality_p'][level_idx] = 1.0
        else:
            results['normality_p'][level_idx] = 1.0

        # 2. Energy fraction (Parseval's theorem: energy in this detail band)
        detail_energy = np.sum(d ** 2) / (2 ** (level_idx + 1))
        results['energy_frac'][level_idx] = detail_energy / (total_energy + 1e-12)

        # 3. Shannon entropy of normalised coefficient distribution
        # High entropy = diverse dynamics; low entropy = near-silent or pure noise
        d_abs = np.abs(d) + 1e-12
        d_norm = d_abs / d_abs.sum()
        results['entropy'][level_idx] = -np.sum(d_norm * np.log2(d_norm + 1e-12))

        # 4. Coefficient standard deviation
        results['coeff_std'][level_idx] = d.std()

        # 5. Zero crossing rate (proportional to dominant frequency)
        zc = np.sum(np.diff(np.sign(d)) != 0) / len(d)
        results['zero_crossing'][level_idx] = zc

        # 6. SNR estimate: ratio of signal variance to MAD-estimated noise
        # MAD-based noise estimate (Donoho 1994)
        delta = np.median(np.abs(d)) / 0.6745
        signal_variance = d.var()
        noise_variance  = delta ** 2
        snr = 10 * np.log10((signal_variance / (noise_variance + 1e-12)) + 1e-12)
        results['snr_estimate'][level_idx] = snr

    return results


def compute_oscillation_cycles(max_level, fs, segment_duration=10):
    """
    For each detail level, compute how many complete oscillation cycles
    fit in segment_duration seconds.
    Minimum cycles needed for reliable attractor reconstruction: 5-10
    """
    cycles = np.zeros(max_level)
    for level in range(1, max_level + 1):
        f_low  = fs / (2 ** (level + 1))
        f_high = fs / (2 ** level)
        f_centre = (f_low + f_high) / 2
        cycles[level - 1] = f_centre * segment_duration
    return cycles


# ══════════════════════════════════════════════════════════════════════════════
# Run analysis
# ══════════════════════════════════════════════════════════════════════════════

print("Computing per-level statistics...")
print(f"Pre-AF records: {len(PREAF_RECORDS)}, NSR records: {len(NSR_RECORDS)}")

# Aggregate results across all records
def aggregate_stats(records, label):
    all_stats = {k: [] for k in ['normality_p', 'energy_frac', 'entropy',
                                   'coeff_std', 'zero_crossing', 'snr_estimate']}
    for rec_id in tqdm(records, desc=label):
        try:
            stats_rec = compute_level_stats(rec_id)
            for k in all_stats:
                all_stats[k].append(stats_rec[k])
        except Exception as e:
            print(f"  [WARN] {rec_id}: {e}")
    # Convert to arrays: (n_records, n_levels)
    return {k: np.array(v) for k, v in all_stats.items()}

preaf_stats = aggregate_stats(PREAF_RECORDS, 'Pre-AF')
nsr_stats   = aggregate_stats(NSR_RECORDS,   'NSR')

levels    = np.arange(1, MAX_LEVEL + 1)
cycles    = compute_oscillation_cycles(MAX_LEVEL, FS_TARGET)

# Frequency bands per level
freq_bands = []
for lv in levels:
    f_low  = FS_TARGET / (2 ** (lv + 1))
    f_high = FS_TARGET / (2 ** lv)
    freq_bands.append(f'{f_low:.1f}–{f_high:.1f} Hz')


# ══════════════════════════════════════════════════════════════════════════════
# Print table
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*100}")
print(f"{'Level':5s}  {'Freq Band':15s}  {'Cycles/10s':10s}  "
      f"{'Normality p':12s}  {'Energy%':8s}  {'Entropy':8s}  "
      f"{'SNR(dB)':8s}  {'Interpretation'}")
print(f"{'='*100}")

for i, lv in enumerate(levels):
    pa_norm  = preaf_stats['normality_p'][:, i].mean()
    ns_norm  = nsr_stats['normality_p'][:, i].mean()
    pa_ener  = preaf_stats['energy_frac'][:, i].mean() * 100
    ns_ener  = nsr_stats['energy_frac'][:, i].mean() * 100
    pa_entr  = preaf_stats['entropy'][:, i].mean()
    pa_snr   = preaf_stats['snr_estimate'][:, i].mean()

    # Interpretation flags
    flags = []
    if pa_norm > 0.05:
        flags.append('NOISE-ONLY (threshold safe)')
    else:
        flags.append('signal present')
    if cycles[i] < 10:
        flags.append('FEW CYCLES (RP unreliable)')
    if pa_ener < 0.5:
        flags.append('low energy')

    print(f"  D{lv:2d}  {freq_bands[i]:15s}  {cycles[i]:10.1f}  "
          f"{pa_norm:12.4f}  {pa_ener:8.2f}%  {pa_entr:8.2f}  "
          f"{pa_snr:8.2f}  {', '.join(flags)}")

print(f"{'='*100}")

# Denoising recommendation
denoise_cutoff = None
for i in range(MAX_LEVEL - 1, -1, -1):
    pa_norm = preaf_stats['normality_p'][:, i].mean()
    if pa_norm < 0.05:  # non-normal = signal still present
        denoise_cutoff = levels[i]
        break

if denoise_cutoff:
    print(f"\nRecommended DENOISING level: {denoise_cutoff}")
    print(f"  Rationale: D{denoise_cutoff} ({freq_bands[denoise_cutoff-1]}) is the "
          f"deepest level with non-Gaussian coefficients (signal present).")
    print(f"  Denoising beyond level {denoise_cutoff} would threshold real signal.")

# Decomposition recommendation
print(f"\nRecommended DECOMPOSITION levels for subband UTRPs:")
for i, lv in enumerate(levels):
    pa_norm  = preaf_stats['normality_p'][:, i].mean()
    pa_ener  = preaf_stats['energy_frac'][:, i].mean() * 100
    cyc      = cycles[i]
    # Good subband: signal present + enough cycles + meaningful energy
    if pa_norm < 0.05 and cyc >= 20 and pa_ener > 0.5:
        print(f"  ✓ D{lv} ({freq_bands[i]}): normality p={pa_norm:.4f}, "
              f"{cyc:.0f} cycles, energy={pa_ener:.1f}%")
    elif pa_norm < 0.05 and cyc >= 10 and pa_ener > 0.5:
        print(f"  ~ D{lv} ({freq_bands[i]}): borderline — {cyc:.0f} cycles "
              f"(min 20 recommended), energy={pa_ener:.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# Figure
# ══════════════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(18, 14))
gs  = gridspec.GridSpec(3, 2, hspace=0.4, wspace=0.35)

x = levels
xlab = [f'D{lv}\n{freq_bands[i].split("–")[0]}Hz' for i, lv in enumerate(levels)]

# ── Panel 1: Normality p-value ─────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
pa_norm_mean = preaf_stats['normality_p'].mean(axis=0)
ns_norm_mean = nsr_stats['normality_p'].mean(axis=0)
ax1.plot(x, pa_norm_mean, 'r-o', label='Pre-AF', markersize=5)
ax1.plot(x, ns_norm_mean, 'b-o', label='NSR',    markersize=5)
ax1.axhline(0.05, color='green', ls='--', lw=2,
            label='α=0.05 (noise threshold)')
ax1.fill_between(x, 0.05, 1.0, alpha=0.08, color='green',
                 label='Gaussian (safe to threshold)')
ax1.set_xlabel('Detail Level')
ax1.set_ylabel('Normality test p-value')
ax1.set_title('Coefficient Normality per Level\n'
              'p > 0.05 → Gaussian noise → safe to threshold')
ax1.legend(fontsize=8)
ax1.set_xticks(x); ax1.set_xticklabels(xlab, fontsize=7)
ax1.grid(alpha=0.3)
ax1.set_ylim(-0.05, 1.05)

# Shade the recommended denoising region
if denoise_cutoff:
    ax1.axvspan(denoise_cutoff + 0.5, MAX_LEVEL + 0.5,
                alpha=0.1, color='green', label='Threshold this region')

# ── Panel 2: Energy fraction ───────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
pa_ener_mean = preaf_stats['energy_frac'].mean(axis=0) * 100
ns_ener_mean = nsr_stats['energy_frac'].mean(axis=0) * 100
pa_ener_std  = preaf_stats['energy_frac'].std(axis=0) * 100
ns_ener_std  = nsr_stats['energy_frac'].std(axis=0) * 100

ax2.bar(x - 0.2, pa_ener_mean, 0.35, yerr=pa_ener_std,
        color='red',  alpha=0.7, label='Pre-AF', capsize=3)
ax2.bar(x + 0.2, ns_ener_mean, 0.35, yerr=ns_ener_std,
        color='blue', alpha=0.7, label='NSR', capsize=3)
ax2.set_xlabel('Detail Level')
ax2.set_ylabel('Energy fraction (%)')
ax2.set_title('Signal Energy per Level\n'
              '(Parseval decomposition)')
ax2.legend(fontsize=8)
ax2.set_xticks(x); ax2.set_xticklabels(xlab, fontsize=7)
ax2.grid(alpha=0.3, axis='y')

# ── Panel 3: Shannon entropy ───────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
pa_entr_mean = preaf_stats['entropy'].mean(axis=0)
ns_entr_mean = nsr_stats['entropy'].mean(axis=0)
pa_entr_std  = preaf_stats['entropy'].std(axis=0)
ns_entr_std  = nsr_stats['entropy'].std(axis=0)

ax3.errorbar(x - 0.1, pa_entr_mean, yerr=pa_entr_std,
             fmt='r-o', label='Pre-AF', markersize=5, capsize=3)
ax3.errorbar(x + 0.1, ns_entr_mean, yerr=ns_entr_std,
             fmt='b-o', label='NSR',    markersize=5, capsize=3)
ax3.set_xlabel('Detail Level')
ax3.set_ylabel('Shannon entropy (bits)')
ax3.set_title('Coefficient Entropy per Level\n'
              'Higher entropy = more diverse dynamics')
ax3.legend(fontsize=8)
ax3.set_xticks(x); ax3.set_xticklabels(xlab, fontsize=7)
ax3.grid(alpha=0.3)

# ── Panel 4: SNR estimate ──────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
pa_snr_mean = preaf_stats['snr_estimate'].mean(axis=0)
ns_snr_mean = nsr_stats['snr_estimate'].mean(axis=0)
pa_snr_std  = preaf_stats['snr_estimate'].std(axis=0)
ns_snr_std  = nsr_stats['snr_estimate'].std(axis=0)

ax4.errorbar(x - 0.1, pa_snr_mean, yerr=pa_snr_std,
             fmt='r-o', label='Pre-AF', markersize=5, capsize=3)
ax4.errorbar(x + 0.1, ns_snr_mean, yerr=ns_snr_std,
             fmt='b-o', label='NSR',    markersize=5, capsize=3)
ax4.axhline(0, color='gray', ls='--', lw=1, label='SNR=0 dB (noise floor)')
ax4.set_xlabel('Detail Level')
ax4.set_ylabel('SNR estimate (dB)')
ax4.set_title('Signal-to-Noise Ratio per Level\n'
              '(Donoho MAD estimator)')
ax4.legend(fontsize=8)
ax4.set_xticks(x); ax4.set_xticklabels(xlab, fontsize=7)
ax4.grid(alpha=0.3)

# ── Panel 5: Oscillation cycles per level ─────────────────────────────────
ax5 = fig.add_subplot(gs[2, 0])
ax5.bar(x, cycles, color='steelblue', alpha=0.8)
ax5.axhline(20, color='green',  ls='--', lw=2,
            label='20 cycles (recommended min for RP)')
ax5.axhline(10, color='orange', ls='--', lw=2,
            label='10 cycles (absolute min for RP)')
ax5.axhline(5,  color='red',    ls='--', lw=2,
            label='5 cycles (insufficient)')
ax5.set_xlabel('Detail Level')
ax5.set_ylabel('Oscillation cycles in 10s segment')
ax5.set_title('Oscillation Cycles per Level\n'
              'Minimum cycles needed for reliable phase-space reconstruction')
ax5.legend(fontsize=8)
ax5.set_xticks(x); ax5.set_xticklabels(xlab, fontsize=7)
ax5.grid(alpha=0.3, axis='y')
ax5.set_yscale('log')

# ── Panel 6: Pre-AF vs NSR entropy difference (discriminability) ───────────
ax6 = fig.add_subplot(gs[2, 1])
entr_diff      = pa_entr_mean - ns_entr_mean
entr_diff_se   = np.sqrt(pa_entr_std**2 + ns_entr_std**2) / np.sqrt(15)
colors = ['red' if d > 0 else 'blue' for d in entr_diff]

bars = ax6.bar(x, entr_diff, color=colors, alpha=0.7)
ax6.errorbar(x, entr_diff, yerr=entr_diff_se, fmt='none',
             color='black', capsize=3)
ax6.axhline(0, color='black', lw=1)
ax6.set_xlabel('Detail Level')
ax6.set_ylabel('Entropy difference (Pre-AF − NSR)')
ax6.set_title('Per-Level Discriminability\n'
              'Red=Pre-AF more complex, Blue=NSR more complex')
ax6.set_xticks(x); ax6.set_xticklabels(xlab, fontsize=7)
ax6.grid(alpha=0.3, axis='y')

# Add frequency band labels on top
for bar, fb in zip(bars, freq_bands):
    ypos = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2, ypos,
             fb, ha='center', va='bottom', fontsize=5, rotation=45)

plt.suptitle(f'SWT Level Selection Analysis — {WAVELET} wavelet, {FS_TARGET} Hz\n'
             f'N={len(PREAF_RECORDS)} Pre-AF + {len(NSR_RECORDS)} NSR records',
             fontsize=13, fontweight='bold')

out = OUT_DIR / 'level_selection_analysis.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nFigure saved → {out}")
print(f"Table saved to stdout above.")
print(f"\nThesis justification summary:")
print(f"  Denoising level: selected as deepest level where detail coefficients")
print(f"  are non-Gaussian (signal present). Levels beyond this contain only")
print(f"  Gaussian noise and can be safely thresholded (Donoho & Johnstone, 1994).")
print(f"  Decomposition level: selected as deepest level satisfying both")
print(f"  (a) ≥20 oscillation cycles per 10s segment (sufficient for phase-space")
print(f"  reconstruction) and (b) non-Gaussian coefficients (signal present).")