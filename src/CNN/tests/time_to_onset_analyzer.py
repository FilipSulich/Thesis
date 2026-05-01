"""
Time-to-Onset Analyzer - Evaluates model performance across different time windows before AF onset
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.CNN.tests.model_evaluator import ModelEvaluator

def _get_predictions(evaluator: ModelEvaluator,
                     data_dir: Path) -> Tuple[np.ndarray, np.ndarray,
                                               np.ndarray, np.ndarray]:
    """Run inference on test split."""
    with h5py.File(data_dir / 'test.h5', 'r') as f:
        labels = f['metadata/labels'][:].astype(np.int64)
        rids   = np.array([r.decode() for r in f['metadata/record_ids'][:]])
        key    = ('time_to_onset_min' if 'time_to_onset_min' in f['metadata']
                  else 'time_to_onset_sec')
        t2o    = f[f'metadata/{key}'][:].astype(np.float32)
        # Convert seconds → minutes
        t2o_min = t2o / 60.0

    from src.CNN.dataset.dataset import UTRPDataset
    ds_mode = 'both' if evaluator.model_arch == 'hybrid' else evaluator.model_arch
    test_ds = UTRPDataset(data_dir, split='test', mode=ds_mode, augment=False)
    loader  = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    all_probs = []
    evaluator.model.eval()
    with torch.no_grad():
        for batch in loader:
            if evaluator.model_arch == 'hybrid':
                x_s, x_m, _ = batch
                logits = evaluator.model(x_s.to(evaluator.device),
                                         x_m.to(evaluator.device))
            else:
                x, _ = batch
                logits = evaluator.model(x.to(evaluator.device))
            probs = torch.softmax(logits, 1)[:, 1].cpu().numpy()
            all_probs.extend(probs)

    return np.array(all_probs), labels, rids, t2o_min

def run_time_to_onset_analysis(
    evaluators: Dict[str, ModelEvaluator],
    data_dir: Path,
    window_size_min: float = 5.0,
    save_plot_path: Optional[Path] = None,
    save_json_path: Optional[Path] = None,
) -> Dict:

    all_results = {}

    for model_name, evaluator in evaluators.items():
        probs, labels, rids, t2o_min = _get_predictions(evaluator, data_dir)

        mask_podd = t2o_min >= 0
        mask_nsr  = labels == 0

        t2o_podd   = t2o_min[mask_podd]
        probs_podd = probs[mask_podd]
        probs_nsr  = probs[mask_nsr]
        labels_nsr = labels[mask_nsr]   # all zeros

        print(f"\n{model_name}:")
        print(f"  p-odd segments: {mask_podd.sum()}")
        print(f"  NSR segments:   {mask_nsr.sum()}")
        print(f"  Time range:     {t2o_podd.min():.1f} – "
              f"{t2o_podd.max():.1f} min before onset")

        bin_edges = np.arange(0, 30 + window_size_min, window_size_min)
        bin_aucs, bin_auprcs, bin_ns, bin_mids = [], [], [], []

        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask_win = (t2o_podd >= lo) & (t2o_podd < hi)
            n_win    = mask_win.sum()
            if n_win == 0:
                continue

            # Combine this window's p-odd segments with all NSR segments
            win_probs  = np.concatenate([probs_podd[mask_win], probs_nsr])
            win_labels = np.concatenate([np.ones(n_win), labels_nsr])

            auc   = float(roc_auc_score(win_labels, win_probs))
            auprc = float(average_precision_score(win_labels, win_probs))
            accuracy = float(np.ac)

            bin_aucs.append(auc)
            bin_auprcs.append(auprc)
            bin_ns.append(int(n_win))
            bin_mids.append(float((lo + hi) / 2.0))

        all_results[model_name] = {
            'bin_mids':  bin_mids,
            'bin_aucs':  bin_aucs,
            'bin_auprcs': bin_auprcs,
            'bin_ns':    bin_ns,
        }

        print(f"  {'Window':<12} {'N PreAF':>8} {'AUC':>8} {'AUPRC':>8}")
        print(f"  {'-'*40}")
        for mid, auc, auprc, n in zip(bin_mids, bin_aucs, bin_auprcs, bin_ns):
            lo_m = mid - window_size_min / 2
            hi_m = mid + window_size_min / 2
            print(f"  {lo_m:.0f}–{hi_m:.0f} min   {n:>8} {auc:>8.4f} {auprc:>8.4f}")

    # ── Plot ───────────────────────────────────────────────────────────────
    colour_map  = {'Multi-scale': '#d62728', 'Single-scale': '#1f77b4'}
    colour_list = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for i, (model_name, res) in enumerate(all_results.items()):
        col = colour_map.get(model_name, colour_list[i % len(colour_list)])
        x   = np.array(res['bin_mids'])

        ax1.plot(x, res['bin_aucs'],   'o-', color=col, lw=2,
                 markersize=6, label=model_name)
        ax2.plot(x, res['bin_auprcs'], 'o-', color=col, lw=2,
                 markersize=6, label=model_name)

    for ax, metric, baseline in [
        (ax1, 'AUC-ROC', 0.5),
        (ax2, 'AUPRC',   None),
    ]:
        ax.axhline(baseline if baseline else
                   # AUPRC random baseline = positive rate in each window
                   # approximate: use overall positive rate
                   (mask_podd.sum() / (mask_podd.sum() + mask_nsr.sum())),
                   color='gray', linestyle='--', lw=1.2,
                   label='Random baseline')
        ax.set_xlabel('Time before AF onset (min)', fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f'{metric} vs time to onset', fontsize=12)
        ax.set_xlim(0, 30)
        ax.set_ylim(0, 1)
        ax.invert_xaxis()
        ax.set_xticks(np.arange(0, 31, 5))
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Classification performance vs time to AF onset\n'
                 '(test split, p-odd vs all NSR segments)', fontsize=12)
    plt.tight_layout()

    if save_plot_path:
        Path(save_plot_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_plot_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved → {save_plot_path}")
    plt.show()
    plt.close()

    if save_json_path:
        Path(save_json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_json_path, 'w') as f:
            json.dump(all_results, f, indent=2)

    return all_results

if __name__ == '__main__':
    DATA_DIR    = Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/data/data_final/utrp_dataset_db8_denoise')
    RESULTS_DIR = Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/data/results_new/time_to_onset_analysis')

    # Load multi-scale model
    eval_multi = ModelEvaluator(
        Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/data/results_new/multi/final_checkpoint.pt'),
        DATA_DIR, 'multi'
    )
    eval_multi.load_model(strict=True, attention_mode='uniform')

    # Load single-scale model (denoised pipeline)
    eval_single = ModelEvaluator(
        Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/data/results_new2/single/final_checkpoint.pt'),
        DATA_DIR, 'single'
    )
    eval_single.load_model(strict=True, attention_mode='trained')

    run_time_to_onset_analysis(
        evaluators={
            'Multi-scale': eval_multi,
            'Single-scale': eval_single,
        },
        data_dir=DATA_DIR,
        window_size_min=5.0,
        save_plot_path=RESULTS_DIR / 'time_to_onset_comparison.png',
        save_json_path=RESULTS_DIR / 'time_to_onset_results.json',
    )