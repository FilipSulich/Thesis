from pathlib import Path
import numpy as np
import h5py
import matplotlib.pyplot as plt
from typing import Optional

# Try to import dataset generator constants and helper functions for preprocessing
try:
    from src.CNN.dataset.dataset_generator import (
        DATA_DIR, FS_ORIG, FS_TARGET, SEGMENT_SAMPLES, N_SEGMENTS, SUBBAND_NAMES,
        denoise_and_extract_subbands, create_utrp as _create_utrp, _minmax_norm, _pad, TAU_SINGLE, M_SINGLE, UTRP_SIZE,
    )
    # wrap imported create_utrp to match expected signature
    def create_utrp(signal, tau, m, output_size=256):
        return _create_utrp(signal, tau, m, output_size)
except Exception:
    DATA_DIR = None
    FS_ORIG = 128
    FS_TARGET = 500
    SEGMENT_SAMPLES = 5000
    N_SEGMENTS = 15
    SUBBAND_NAMES = [f'D{i}' for i in range(2, 9)]

    def denoise_and_extract_subbands(x):
        return x[:SEGMENT_SAMPLES], {name: x[:SEGMENT_SAMPLES] for name in SUBBAND_NAMES}

    def create_utrp(signal, tau, m, output_size=256):
        # naive distance matrix -> jet rgb (fallback)
        sig = signal[:SEGMENT_SAMPLES]
        n = len(sig) - (m - 1) * tau
        if n <= 0:
            return np.zeros((3, output_size, output_size), dtype=np.float32), np.zeros((output_size, output_size), dtype=np.float32)
        embedded = np.array([sig[i:i + m * tau:tau] for i in range(n)])
        diff = embedded[:, None, :] - embedded[None, :, :]
        distances = np.sqrt(np.sum(diff ** 2, axis=2)).astype(np.float32)
        d_max = distances.max() if distances.size else 1.0
        distances = distances / (d_max + 1e-9)
        try:
            import cv2
            dist_r = cv2.resize(distances, (output_size, output_size), interpolation=cv2.INTER_LINEAR).astype(np.float32)
            u8 = (dist_r * 255).astype(np.uint8)
            bgr = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            return rgb.transpose(2, 0, 1), dist_r
        except Exception:
            # very small fallback: grayscale mapping
            from skimage.transform import resize
            dist_r = resize(distances, (output_size, output_size), order=1, preserve_range=True).astype(np.float32)
            rgb = np.stack([dist_r, dist_r, dist_r], axis=0)
            return rgb, dist_r


class UTRPVisualizer:
    def __init__(self, hdf5_path: str):
        self.hdf5_path = Path(hdf5_path)
        self.h5 = h5py.File(self.hdf5_path, 'r')

    def close(self):
        try:
            self.h5.close()
        except Exception:
            pass

    def _get_sample_info(self, idx: int):
        # pull single-scale, distance (optional), multi-scale, metadata
        ss = self.h5['single_scale/utrps'][idx]
        ms = self.h5['multi_scale/utrps'][idx]
        labels = self.h5['metadata/labels']
        record_ids = self.h5['metadata/record_ids']
        seg_idxs = self.h5['metadata/segment_indices']
        label = labels[idx] if idx < len(labels) else None
        record_id = record_ids[idx].decode('utf-8') if isinstance(record_ids[idx], bytes) else record_ids[idx]
        segment_idx = int(seg_idxs[idx]) if idx < len(seg_idxs) else 0
        # distance matrix may not be stored
        dist = None
        # return single-scale RGB (C,H,W), dist, multi-scale (C_total,H,W), label, record_id, segment_idx
        return ss, dist, ms, label, record_id, segment_idx

    def show_sample(self, idx: int, include_raw_signal: bool = False, save_path: Optional[Path] = None,
                    crop_center_seconds: Optional[float] = None, vertical: bool = True):
        """Create a figure visualizing the UTRP sample.

        Creates two separate outputs (signal / subbands) and (RPs). If save_path is
        provided two files will be written with suffixes '_signal' and '_rps'.

        Additional options:
          - crop_center_seconds: if provided, keep only this many seconds from the
            center of reconstructed signals/subbands (useful to remove edge artifacts).
          - vertical: if True, stack the main RP panels vertically instead of side-by-side.
        """

        ss, dist, ms, label, record_id, segment_idx = self._get_sample_info(idx)

        # Prepare multi-scale RGB images: ms shape (C,H,W) -> group by 3 -> (n_subbands,3,H,W)
        n_channels = ms.shape[0]
        group_size = 3
        n_subbands = n_channels // group_size
        ms_rgbs = ms.reshape(n_subbands, group_size, ms.shape[1], ms.shape[2])

        # attempt to reconstruct signal + subbands from raw record
        reconstructed = False
        subband_signals = None
        seg_norm = None
        ss_utrp = None
        ss_dist = None

        if DATA_DIR is not None and record_id is not None:
            try:
                import wfdb
                from scipy import signal as sp_signal

                rec = wfdb.rdrecord(str(Path(DATA_DIR) / record_id))
                raw = rec.p_signal[:, 0].astype(np.float64)
                n_tgt = int(len(raw) * FS_TARGET / FS_ORIG)
                res = sp_signal.resample(raw, n_tgt)
                max_start = len(res) - SEGMENT_SAMPLES
                if max_start > 0:
                    starts = np.linspace(0, max_start, N_SEGMENTS, dtype=int)
                    seg = res[starts[segment_idx]:starts[segment_idx] + SEGMENT_SAMPLES]
                    seg_norm = _minmax_norm(seg.astype(np.float32))
                    seg_pad = _pad(seg_norm)
                    seg_den, subbands = denoise_and_extract_subbands(seg_pad)

                    # Optionally crop the center to avoid edge artifacts
                    if crop_center_seconds is not None and crop_center_seconds > 0:
                        keep_samples = int(round(crop_center_seconds * FS_TARGET))
                        if keep_samples < 1:
                            keep_samples = 1
                        if keep_samples < len(seg_den):
                            start = (len(seg_den) - keep_samples) // 2
                            seg_den = seg_den[start:start + keep_samples]
                            # crop each subband
                            for k, v in list(subbands.items()):
                                arr = np.asarray(v)
                                if len(arr) >= keep_samples:
                                    subbands[k] = arr[start:start + keep_samples]
                                else:
                                    pad = keep_samples - len(arr)
                                    subbands[k] = np.pad(arr, (pad // 2, pad - pad // 2), mode='reflect')

                    ss_utrp, ss_dist = create_utrp(seg_den, TAU_SINGLE, M_SINGLE, 256)
                    reconstructed = True
                    subband_signals = subbands
            except Exception as e:
                print('Could not reconstruct from raw records:', e)

        # determine save paths
        signal_path = None
        rps_path = None
        if save_path is not None:
            spath = Path(save_path)
            signal_path = spath.with_name(spath.stem + '_signal' + spath.suffix)
            rps_path = spath.with_name(spath.stem + '_rps' + spath.suffix)

        # FIGURE 1: signal (and subbands). Plot signal and stack subbands vertically to avoid clustering
        fig_sig = plt.figure(figsize=(14, 3 if not reconstructed else max(3, 0.4 * (1 + len(SUBBAND_NAMES)))))
        if reconstructed and seg_norm is not None:
            ax_sig = fig_sig.add_subplot(1, 1, 1)
            to_plot = seg_den if 'seg_den' in locals() else seg_norm
            ax_sig.plot(to_plot, lw=0.6)
            ax_sig.set_title(f'Resampled ECG (record={record_id}, seg={segment_idx})')
            ax_sig.set_xticks([])

            # subband waveforms in separate figure (vertical stack)
            fig_sb = plt.figure(figsize=(14, max(2, 0.8 * len(SUBBAND_NAMES))))
            n_sb = len(SUBBAND_NAMES)
            for i, name in enumerate(SUBBAND_NAMES):
                ax = fig_sb.add_subplot(n_sb, 1, i + 1)
                s = subband_signals.get(name, np.zeros(SEGMENT_SAMPLES))
                ax.plot(s, lw=0.5)
                ax.set_title(name)
                ax.set_xticks([])
            if signal_path is not None:
                fig_sig.savefig(signal_path.with_name(signal_path.stem + '_full' + signal_path.suffix), dpi=150)
                fig_sb.savefig(signal_path.with_name(signal_path.stem + '_subbands' + signal_path.suffix), dpi=150)
                print('Saved signal figures to', signal_path.with_name(signal_path.stem + '_full' + signal_path.suffix))
            else:
                plt.show(block=False)
                fig_sb.show()
            plt.close(fig_sb)
        else:
            fig_sig.text(0.5, 0.5, 'Raw record not available — see RP figures', ha='center', va='center')
            if signal_path is not None:
                fig_sig.savefig(signal_path.with_name(signal_path.stem + '_full' + signal_path.suffix), dpi=150)
            else:
                plt.show()
        plt.close(fig_sig)

        # FIGURE 2: RPs and distance + multi-scale
        from matplotlib import gridspec
        # vertical stacking: top single-scale, middle distance, bottom multi-scale
        fig_rp = plt.figure(figsize=(12, 14))
        if vertical:
            outer = gridspec.GridSpec(nrows=3, ncols=1, figure=fig_rp, height_ratios=[1, 0.9, 1.6], hspace=0.25)

            ax_rp = fig_rp.add_subplot(outer[0])
            if reconstructed and ss_utrp is not None:
                ax_rp.imshow(ss_utrp.transpose(1, 2, 0))
                ax_rp.set_title('Reconstructed UTRP')
            else:
                ax_rp.imshow(ss.transpose(1, 2, 0))
                ax_rp.set_title('Stored single-scale UTRP')
            ax_rp.axis('off')

            ax_dist = fig_rp.add_subplot(outer[1])
            if reconstructed and ss_dist is not None:
                im = ax_dist.imshow(ss_dist, cmap='viridis')
                ax_dist.set_title('Reconstructed distance matrix')
            elif dist is not None:
                im = ax_dist.imshow(dist, cmap='viridis')
                ax_dist.set_title('Stored distance matrix')
            else:
                ax_dist.text(0.5, 0.5, 'No distance matrix', ha='center', va='center')
            ax_dist.axis('off')
            if 'im' in locals():
                fig_rp.colorbar(im, ax=ax_dist, fraction=0.04, pad=0.02)

            # multi-scale grid in bottom area
            ms_spec = outer[2]
            cols = min(7, n_subbands) if n_subbands > 0 else 1
            rows = (n_subbands + cols - 1) // cols if n_subbands > 0 else 1
            ms_gs = gridspec.GridSpecFromSubplotSpec(rows, cols, subplot_spec=ms_spec, hspace=0.25, wspace=0.25)
            for i in range(n_subbands):
                axm = fig_rp.add_subplot(ms_gs[i])
                if reconstructed and subband_signals is not None:
                    rgb, _ = create_utrp(subband_signals.get(SUBBAND_NAMES[i], np.zeros(SEGMENT_SAMPLES)), TAU_SINGLE, M_SINGLE, 256)
                    axm.imshow(rgb.transpose(1, 2, 0))
                else:
                    rgb = ms_rgbs[i].transpose(1, 2, 0)
                    axm.imshow(np.clip(rgb, 0, 1))
                axm.set_title(SUBBAND_NAMES[i] if i < len(SUBBAND_NAMES) else f'D{i+2}')
                axm.axis('off')
        else:
            # fallback: previous side-by-side layout
            outer = gridspec.GridSpec(nrows=1, ncols=3, figure=fig_rp, width_ratios=[1, 1, 2], wspace=0.25)
            ax_rp = fig_rp.add_subplot(outer[0])
            if reconstructed and ss_utrp is not None:
                ax_rp.imshow(ss_utrp.transpose(1, 2, 0))
                ax_rp.set_title('Reconstructed UTRP')
            else:
                ax_rp.imshow(ss.transpose(1, 2, 0))
                ax_rp.set_title('Stored single-scale UTRP')
            ax_rp.axis('off')

            ax_dist = fig_rp.add_subplot(outer[1])
            if reconstructed and ss_dist is not None:
                im = ax_dist.imshow(ss_dist, cmap='viridis')
                ax_dist.set_title('Reconstructed distance matrix')
            elif dist is not None:
                im = ax_dist.imshow(dist, cmap='viridis')
                ax_dist.set_title('Stored distance matrix')
            else:
                ax_dist.text(0.5, 0.5, 'No distance matrix', ha='center', va='center')
            ax_dist.axis('off')
            if 'im' in locals():
                fig_rp.colorbar(im, ax=ax_dist, fraction=0.046)

            # multi-scale grid
            ms_spec = outer[2]
            cols = min(7, n_subbands) if n_subbands > 0 else 1
            rows = (n_subbands + cols - 1) // cols if n_subbands > 0 else 1
            ms_gs = gridspec.GridSpecFromSubplotSpec(rows, cols, subplot_spec=ms_spec, hspace=0.25, wspace=0.25)
            for i in range(n_subbands):
                axm = fig_rp.add_subplot(ms_gs[i])
                if reconstructed and subband_signals is not None:
                    rgb, _ = create_utrp(subband_signals.get(SUBBAND_NAMES[i], np.zeros(SEGMENT_SAMPLES)), TAU_SINGLE, M_SINGLE, 256)
                    axm.imshow(rgb.transpose(1, 2, 0))
                else:
                    rgb = ms_rgbs[i].transpose(1, 2, 0)
                    axm.imshow(np.clip(rgb, 0, 1))
                axm.set_title(SUBBAND_NAMES[i] if i < len(SUBBAND_NAMES) else f'D{i+2}')
                axm.axis('off')

        if rps_path is not None:
            fig_rp.savefig(rps_path, dpi=150)
            print('Saved RP figure to', rps_path)
        else:
            plt.show()
        plt.close(fig_rp)


if __name__ == '__main__':
    # quick smoke test (saves PNGs for sample 0); crops center 8s to remove edge artifacts
    try:
        viz = UTRPVisualizer('/data/data_new/attention/utrp_new_denoise2.h5')
        viz.show_sample(12, include_raw_signal=True, crop_center_seconds=8.0)
        viz.close()
    except Exception as e:
        print('Smoke test failed:', e)

