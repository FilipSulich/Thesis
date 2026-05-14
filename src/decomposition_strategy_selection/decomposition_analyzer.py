import os
import sys
import numpy as np
import pandas as pd
import json

from src.decomposition_strategy_selection.visualization import DecompositionVisualizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.recurrence_plot.AMI import AMI
from src.recurrence_plot.FNN import FNN
from src.pipelines.pipeline_wavelets import Pipeline
from src.recurrence_plot.recurrence_plot_generation import RecurrencePlotGenerator
from pyrqa.settings import Settings
from pyrqa.analysis_type import Cross
from pyrqa.neighbourhood import Unthresholded
from pyrqa.metric import EuclideanMetric
from pyrqa.computation import RPComputation
from pyrqa.time_series import TimeSeries


class DecompositionAnalyzer:
    """
    Core class for wavelet decomposition and recurrence plot analysis.

    Pipeline comparison methodology:
    - Composite sine is decomposed by each pipeline
    - Each subband is min-max normalised at its native length
    - RQA metrics are computed using ground truth tau, m, and epsilon
    - Structural metrics (Hamming, RP correlation) use zero-padded RPs
      to match ground truth size — padding only affects structural
      comparison, not RQA computation
    """

    def __init__(self, subband_freqs=None, fs=128, n=1024,
                 level=4, wavelet='db4', output_dir=None):
        self.FS      = fs
        self.N       = n
        self.LEVEL   = level
        self.WAVELET = wavelet

        if subband_freqs is None:
            self.SUBBAND_FREQS = {
                'A4': 2, 'D4': 6, 'D3': 12, 'D2': 24, 'D1': 48
            }
        else:
            self.SUBBAND_FREQS = subband_freqs

        self.BAND_NAMES = ['D1', 'D2', 'D3', 'D4', 'A4']
        self.PIPELINES  = ['swt_recon', 'swt_direct', 'dwt_recon']

        if output_dir is None:
            self.OUTPUT_DIR = '/src/decomposition_strategy_selection/results'
        else:
            self.OUTPUT_DIR = output_dir

        self.visualizer = DecompositionVisualizer(
            output_dir=self.OUTPUT_DIR
        )
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Signal generation
    # ------------------------------------------------------------------ #

    def generate_composite_sine(self, random_phase=False):
        """Sum of sinusoids at one frequency per subband."""
        t      = np.arange(self.N) / self.FS
        signal = np.zeros(self.N)
        for band, freq in self.SUBBAND_FREQS.items():
            phase   = np.random.uniform(0, 2 * np.pi) if random_phase else 0.0
            signal += np.sin(2 * np.pi * freq * t + phase)
        return signal

    def generate_single_sine(self, freq, random_phase=False):
        """Single sinusoid at a given frequency."""
        t     = np.arange(self.N) / self.FS
        phase = np.random.uniform(0, 2 * np.pi) if random_phase else 0.0
        return np.sin(2 * np.pi * freq * t + phase)

    # ------------------------------------------------------------------ #
    # Normalisation
    # ------------------------------------------------------------------ #

    def min_max_normalize(self, signal):
        lo, hi = signal.min(), signal.max()
        rng = hi - lo
        return (signal - lo) / rng if rng > 0 else signal.copy()

    # ------------------------------------------------------------------ #
    # Decomposition
    # ------------------------------------------------------------------ #

    def decompose_signal(self, signal, pipeline_name):
        """Decompose signal using the named wavelet pipeline."""
        pipeline_obj = Pipeline(
            pipeline_name='all', level=self.LEVEL, wavelet=self.WAVELET
        )
        func = getattr(pipeline_obj, f"pipeline_{pipeline_name}")
        return func(signal)

    # ------------------------------------------------------------------ #
    # Embedding parameter estimation
    # ------------------------------------------------------------------ #

    @staticmethod
    def get_optimal_params(signal):
        """Estimate tau (AMI first minimum) and m (FNN 5% threshold)."""
        ami = AMI(signal)
        tau = ami.get_first_minimum()
        if tau is None:
            tau = max(1, len(signal) // 100)
            print(f"  Warning: AMI found no minimum — fallback tau={tau}")
        fnn = FNN(signal=signal, tau=tau)
        m   = fnn.get_optimal_dimension()
        return tau, m

    # ------------------------------------------------------------------ #
    # Recurrence plot computation
    # ------------------------------------------------------------------ #

    def get_unthresholded_matrix(self, signal, tau, m):
        """Unthresholded (continuous) distance matrix via pyrqa."""
        ts = TimeSeries(signal, embedding_dimension=m, time_delay=tau)
        settings = Settings(
            ts,
            analysis_type=Cross,
            neighbourhood=Unthresholded(),
            similarity_measure=EuclideanMetric,
            theiler_corrector=1,
        )
        result = RPComputation.create(settings, verbose=False).run()
        return result.recurrence_matrix_reverse_normalized

    def compute_rp_metrics(self, signal, tau, m, epsilon):
        """
        Compute binary RP and RQA metrics (DET, LAM, TT, ENTR, …).
        Returns a dict; values are 0.0 on failure.
        """
        try:
            os.environ['PYOPENCL_CTX'] = '0'
            rp_gen = RecurrencePlotGenerator(
                signal=signal,
                time_delay=tau,
                embedding_dim=m,
                path=self.OUTPUT_DIR,
                epsilon=epsilon,
                unthresholded=False,
            )
            rqa_result, rp_matrix = rp_gen.compute_rqa_binary()

            def _safe(v):
                return float(v) if not np.isnan(v) else 0.0

            return {
                'DET':               _safe(rqa_result.determinism),
                'LAM':               _safe(rqa_result.laminarity),
                'TT':                _safe(rqa_result.trapping_time),
                'ENTR':              _safe(rqa_result.entropy_diagonal_lines),
                'L_mean_diagonal':   _safe(rqa_result.average_diagonal_line),
                'L_longest_diagonal':_safe(rqa_result.longest_diagonal_line),
                'L_longest_vertical':_safe(rqa_result.longest_vertical_line),
                'rp_matrix':         rp_matrix,
            }
        except Exception as e:
            print(f"  Warning: RP computation failed: {e}")
            return {
                'DET': 0.0, 'LAM': 0.0, 'TT': 0.0, 'ENTR': 0.0,
                'L_mean_diagonal': 0.0, 'L_longest_diagonal': 0.0,
                'L_longest_vertical': 0.0, 'rp_matrix': None,
            }

    # ------------------------------------------------------------------ #
    # Structural similarity helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _unwrap_rp(rp_obj):
        """Extract numpy array from pyrqa result object if needed."""
        if hasattr(rp_obj, 'recurrence_matrix_reverse_normalized'):
            return rp_obj.recurrence_matrix_reverse_normalized
        return rp_obj

    @staticmethod
    def _pad_to_size(matrix, target_size):
        """
        Zero-pad a square matrix to target_size × target_size.
        Used only for structural comparison — never for RQA computation.
        """
        s = matrix.shape[0]
        if s == target_size:
            return matrix
        if s > target_size:
            return matrix[:target_size, :target_size]
        padded = np.zeros((target_size, target_size), dtype=matrix.dtype)
        padded[:s, :s] = matrix
        return padded

    def _hamming(self, rp_a, rp_b):
        """Normalised Hamming distance between two same-size binary RPs."""
        a = (rp_a > 0).astype(np.int8)
        b = (rp_b > 0).astype(np.int8)
        return float(np.sum(np.abs(a - b))) / (rp_a.shape[0] ** 2)

    def _correlation(self, rp_a, rp_b):
        """Pearson correlation between two same-size RP matrices."""
        return float(np.corrcoef(rp_a.ravel(), rp_b.ravel())[0, 1])

    # ------------------------------------------------------------------ #
    # Ground truth
    # ------------------------------------------------------------------ #

    def generate_ground_truth(self, save_path=None):
        """
        Generate reference RPs from pure single-frequency sines.

        For each subband:
          1. Generate single sine at subband frequency
          2. Min-max normalise
          3. Estimate tau and m (AMI / FNN)
          4. Compute epsilon = 10% of max distance
          5. Compute binary RP and RQA metrics
        """
        if save_path is None:
            save_path = os.path.join(self.OUTPUT_DIR, 'ground_truth.json')

        print("Generating ground truth...")
        ground_truth      = {}
        ground_truth_data = {}

        for band, freq in self.SUBBAND_FREQS.items():
            print(f"  {band} ({freq} Hz) …")

            pure_sine = self.min_max_normalize(self.generate_single_sine(freq))
            tau, m    = self.get_optimal_params(pure_sine)

            dist_mat    = self.get_unthresholded_matrix(pure_sine, tau, m)
            max_dist    = dist_mat.max()
            epsilon     = 0.1 * max_dist

            metrics = self.compute_rp_metrics(pure_sine, tau, m, epsilon)

            ground_truth[band] = {
                'freq_hz':            freq,
                'tau':                int(tau),
                'm':                  int(m),
                'epsilon':            float(epsilon),
                'DET':                float(metrics['DET']),
                'LAM':                float(metrics['LAM']),
                'TT':                 float(metrics['TT']),
                'ENTR':               float(metrics['ENTR']),
                'L_mean_diagonal':    float(metrics['L_mean_diagonal']),
                'L_longest_diagonal': float(metrics['L_longest_diagonal']),
                'L_longest_vertical': float(metrics['L_longest_vertical']),
            }
            ground_truth_data[band] = {
                'rp_matrix':       metrics['rp_matrix'],
                'distance_matrix': dist_mat,
                'pure_sine':       pure_sine,
            }
            print(f"    tau={tau}  m={m}  eps={epsilon:.4f}  "
                  f"DET={metrics['DET']:.4f}  LAM={metrics['LAM']:.4f}  "
                  f"TT={metrics['TT']:.4f}")

        # Save JSON
        with open(save_path, 'w') as f:
            json.dump(ground_truth, f, indent=2)

        # Save NPZ
        npz_path = save_path.replace('.json', '_data.npz')
        arrays = {}
        for band in self.BAND_NAMES:
            rp = self._unwrap_rp(ground_truth_data[band]['rp_matrix'])
            arrays[f'{band}_rp']     = rp
            arrays[f'{band}_dist']   = ground_truth_data[band]['distance_matrix']
            arrays[f'{band}_signal'] = ground_truth_data[band]['pure_sine']
        np.savez(npz_path, **arrays)

        print(f"\nGround truth saved → {save_path}")
        return ground_truth, ground_truth_data

    def load_ground_truth(self, json_path=None, npz_path=None):
        """Load previously saved ground truth."""
        if json_path is None:
            json_path = os.path.join(self.OUTPUT_DIR, 'ground_truth.json')
        if npz_path is None:
            npz_path  = os.path.join(self.OUTPUT_DIR, 'ground_truth_data.npz')

        with open(json_path) as f:
            ground_truth = json.load(f)

        data              = np.load(npz_path, allow_pickle=True)
        ground_truth_data = {}
        for band in self.BAND_NAMES:
            ground_truth_data[band] = {
                'rp_matrix':       data[f'{band}_rp'],
                'distance_matrix': data[f'{band}_dist'],
                'pure_sine':       data[f'{band}_signal'],
            }
        return ground_truth, ground_truth_data

    # ------------------------------------------------------------------ #
    # Pipeline analysis
    # ------------------------------------------------------------------ #

    def analyze_pipeline(self, signal, pipeline_name,
                         ground_truth=None, ground_truth_data=None):
        """
        Decompose signal with the named pipeline and compute metrics.

        Design principles:
        - Subbands are kept at their native length (no resampling, no padding)
        - RQA metrics use ground truth tau, m, AND epsilon for fair comparison
        - Structural metrics (Hamming, RP correlation) zero-pad the pipeline
          RP to match ground truth size — padding affects only these metrics
        """
        subbands = self.decompose_signal(signal, pipeline_name)
        results  = []

        for band in self.BAND_NAMES:
            freq = self.SUBBAND_FREQS[band]

            # Extract subband at native length
            if isinstance(subbands, (list, np.ndarray)) and len(subbands) == 1:
                subband_signal = subbands[0]
            else:
                subband_signal = subbands[band]

            # Min-max normalise at native length
            subband_signal = self.min_max_normalize(subband_signal)

            # Ground truth embedding parameters AND epsilon
            if ground_truth is not None and band in ground_truth:
                tau     = ground_truth[band]['tau']
                m       = ground_truth[band]['m']
                epsilon = ground_truth[band]['epsilon']
            else:
                tau, m = self.get_optimal_params(subband_signal)
                dm      = self.get_unthresholded_matrix(subband_signal, tau, m)
                epsilon = 0.1 * dm.max()

            # Compute RQA from native-length subband signal
            metrics = self.compute_rp_metrics(subband_signal, tau, m, epsilon)

            # RQA deviations vs ground truth
            det_deviation  = np.nan
            lam_deviation  = np.nan
            tt_deviation   = np.nan
            entr_deviation = np.nan

            if ground_truth is not None and band in ground_truth:
                ref = ground_truth[band]
                det_deviation  = abs(ref['DET']  - metrics['DET'])
                lam_deviation  = abs(ref['LAM']  - metrics['LAM'])
                tt_deviation   = abs(ref['TT']   - metrics['TT'])
                entr_deviation = abs(ref['ENTR'] - metrics['ENTR'])

            # Structural similarity — zero-pad to ground truth RP size
            hamming_distance = np.nan
            rp_pattern_corr  = np.nan

            if (ground_truth_data is not None
                    and band in ground_truth_data
                    and metrics['rp_matrix'] is not None):

                ref_rp  = self._unwrap_rp(ground_truth_data[band]['rp_matrix'])
                pipe_rp = self._unwrap_rp(metrics['rp_matrix'])

                if ref_rp is not None and pipe_rp is not None:
                    try:
                        ref_size         = ref_rp.shape[0]
                        pipe_rp_padded   = self._pad_to_size(pipe_rp, ref_size)
                        hamming_distance = self._hamming(ref_rp, pipe_rp_padded)
                        rp_pattern_corr  = self._correlation(ref_rp, pipe_rp_padded)
                    except Exception as e:
                        print(f"  Warning: Structural comparison failed "
                              f"for {pipeline_name}/{band}: {e}")

            results.append({
                'pipeline':            pipeline_name,
                'band':                band,
                'freq_hz':             freq,
                'signal_length':       len(subband_signal),
                'tau':                 tau,
                'm':                   m,
                'epsilon':             float(epsilon),
                'DET':                 metrics['DET'],
                'LAM':                 metrics['LAM'],
                'TT':                  metrics['TT'],
                'ENTR':                metrics['ENTR'],
                'L_mean_diagonal':     metrics['L_mean_diagonal'],
                'L_longest_diagonal':  metrics['L_longest_diagonal'],
                'L_longest_vertical':  metrics['L_longest_vertical'],
                'det_deviation':       det_deviation,
                'lam_deviation':       lam_deviation,
                'tt_deviation':        tt_deviation,
                'entr_deviation':      entr_deviation,
                'hamming_distance':    hamming_distance,
                'rp_pattern_corr':     rp_pattern_corr,
            })

        return pd.DataFrame(results)

    # ------------------------------------------------------------------ #
    # Visualisation
    # ------------------------------------------------------------------ #

    def visualize_all_pipelines(self, signal=None, ground_truth=None,
                                 ground_truth_data=None, visualizer=None,
                                 random_phase=False):
        """Visualise decomposed subbands and RPs for all pipelines."""
        from scipy.signal import resample as scipy_resample

        if visualizer is None:
            visualizer = DecompositionVisualizer(
                output_dir=self.OUTPUT_DIR,
                subband_freqs=self.SUBBAND_FREQS,
                fs=self.FS,
            )

        if signal is None:
            signal = self.generate_composite_sine(random_phase=random_phase)

        if ground_truth is None or ground_truth_data is None:
            gt_path = os.path.join(self.OUTPUT_DIR, 'ground_truth.json')
            if os.path.exists(gt_path):
                ground_truth, ground_truth_data = self.load_ground_truth(gt_path)
            else:
                ground_truth, ground_truth_data = self.generate_ground_truth(gt_path)

        all_pipeline_data = {}

        # Ground truth
        gt_subbands = {b: ground_truth_data[b]['pure_sine']  for b in self.BAND_NAMES}
        gt_rps      = {b: ground_truth_data[b]['rp_matrix']  for b in self.BAND_NAMES}
        gt_tau      = {b: ground_truth[b]['tau']              for b in self.BAND_NAMES}
        gt_m        = {b: ground_truth[b]['m']                for b in self.BAND_NAMES}

        visualizer.plot_decomposition_with_rps(
            signal=self.generate_composite_sine(random_phase=False),
            decomposed_subbands=gt_subbands,
            rp_matrices=gt_rps,
            pipeline_name='ground_truth',
            tau_values=gt_tau,
            m_values=gt_m,
        )
        all_pipeline_data['ground_truth'] = {
            'subbands': gt_subbands, 'rps': gt_rps,
            'tau': gt_tau, 'm': gt_m,
        }

        for pipeline in self.PIPELINES:
            print(f"\nVisualising {pipeline} …")
            subbands = self.decompose_signal(signal, pipeline)

            pipeline_subbands, pipeline_rps = {}, {}
            pipeline_tau, pipeline_m        = {}, {}

            for band in self.BAND_NAMES:
                if isinstance(subbands, (list, np.ndarray)) and len(subbands) == 1:
                    sub = subbands[0]
                else:
                    sub = subbands[band]

                # Resample to full length only for visualisation
                # (so all plots have the same x-axis scale)
                if len(sub) != self.N:
                    sub = scipy_resample(sub, self.N)

                sub = self.min_max_normalize(sub)

                tau     = ground_truth[band]['tau']
                m_val   = ground_truth[band]['m']
                epsilon = ground_truth[band]['epsilon']

                metrics = self.compute_rp_metrics(sub, tau, m_val, epsilon)

                pipeline_subbands[band] = sub
                pipeline_rps[band]      = metrics['rp_matrix']
                pipeline_tau[band]      = tau
                pipeline_m[band]        = m_val

            visualizer.plot_decomposition_with_rps(
                signal=signal,
                decomposed_subbands=pipeline_subbands,
                rp_matrices=pipeline_rps,
                pipeline_name=pipeline,
                tau_values=pipeline_tau,
                m_values=pipeline_m,
            )
            all_pipeline_data[pipeline] = {
                'subbands': pipeline_subbands, 'rps': pipeline_rps,
                'tau': pipeline_tau, 'm': pipeline_m,
            }

        return all_pipeline_data
