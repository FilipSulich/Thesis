import os
import sys
import h5py
from joblib import Parallel, delayed
from matplotlib import gridspec
from sklearn.decomposition import PCA
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lorenz_solver.synthetic import SyntheticDatasetGenerator
from lorenz_solver.lorenz import LorenzSolver
from pipelines.pipeline_wavelets import Pipeline
from recurrence_plot.AMI import AMI
from recurrence_plot.FNN import FNN
from recurrence_plot.recurrence_plot_generation import RecurrencePlotGenerator

from scipy import stats
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_score
class ParallelExperiment:
    def __init__(self, n_jobs=-1):

        self.SIGNAL_LENGTH = 1000
        self.N_SAMPLES_PER_CLASS = 100
        self.SAMPLING_RATE = 128

        self.WAVELET = 'db4'
        self.LEVEL = 4

        self.MAX_LAG_AMI = 50
        self.MAX_DIM_FNN = 10

        self.FNN_THRESHOLD = 5.0
        self.TARGET_RR = 0.05

        self.SNR_LEVELS = [30, 20, 10, 5]
        self.SHIFT_RANGE = 30

        self.n_jobs = n_jobs

        self.BASE_DIR = '/Users/filipsulich/Developer/uni/Thesis/Thesis'
        self.OUTPUT_DIR = os.path.join(self.BASE_DIR, 'outputs/experiment1_synthetic')
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

        self.DECOMPOSED_PATH = os.path.join(self.OUTPUT_DIR, 'decomposed_signals_sines.h5')

        self.PIPELINES = ['dwt_recon', 'dwt_direct', 'swt_recon', 'swt_direct']

        self.PIPELINE_LABELS = {
            'single_scale': 'Single-Scale (Baseline)',
            'dwt_recon': 'DWT Recon', 'dwt_direct': 'DWT Coeff',
            'swt_recon': 'SWT Recon', 'swt_direct': 'SWT Coeff'
        }

        self.RQA_METRICS = ['RR', 'DET', 'LAM', 'TT', 'ENTR']
        self.BAND_NAMES = ['D1', 'D2', 'D3', 'D4', 'A4']

        self.synthetic_data_generator = SyntheticDatasetGenerator(length=self.SIGNAL_LENGTH, samples_per_class=self.N_SAMPLES_PER_CLASS, fs=self.SAMPLING_RATE)

    def decompose_and_save(self, X, y):
        if os.path.exists(self.DECOMPOSED_PATH): return
        pipeline_obj = Pipeline(pipeline_name='all', level=self.LEVEL, wavelet=self.WAVELET)

        with h5py.File(self.DECOMPOSED_PATH, 'w') as f:
            f.create_dataset('signals', data=X)
            f.create_dataset('labels', data=y)

            for p_name in self.PIPELINES:
                grp = f.create_group(p_name)
                func = getattr(pipeline_obj, f"pipeline_{p_name}")
                results = Parallel(n_jobs=self.n_jobs)(delayed(func)(sig) for sig in X)

                for band in self.BAND_NAMES:
                    band_data = []
                    for res in results:
                        sig = res[band]
                        if len(sig) >= self.SIGNAL_LENGTH:
                            band_data.append(sig[:self.SIGNAL_LENGTH])
                        else:
                            band_data.append(np.pad(sig, (0, self.SIGNAL_LENGTH - len(sig)), mode='wrap'))
                    grp.create_dataset(band, data=np.array(band_data), dtype='float32')

    @staticmethod
    def _compute_rqa_worker(signal, params, target_rr, metrics_list, output_dir):
        try:
            sig_norm = (signal - signal.mean()) / (signal.std() + 1e-10)
            ami = AMI(sig_norm, max_lag=params['max_lag'])
            tau = ami.get_first_minimum() or 10
            try:
                fnn = FNN(sig_norm, tau=tau, max_dim=params['max_dim'])
                m = fnn.get_optimal_dimension(threshold=params['fnn_thresh']) or 3
            except:
                m = 3

            from scipy.spatial.distance import pdist, squareform
            n = len(signal) - (m - 1) * tau
            if n < 10: return [0.0] * len(metrics_list)
            embedded = np.array([signal[i:i + m * tau:tau] for i in range(n)])
            distances = squareform(pdist(embedded, metric='euclidean'))
            all_dists = distances[np.triu_indices_from(distances, k=1)]
            eps_min, eps_max = np.percentile(all_dists, 1), np.percentile(all_dists, 20)
            eps_mid = (eps_min + eps_max) / 2
            for _ in range(20):
                eps_mid = (eps_min + eps_max) / 2
                rp_test = (distances <= eps_mid).astype(int)
                np.fill_diagonal(rp_test, 0)
                rr = np.sum(rp_test) / (rp_test.shape[0] * (rp_test.shape[0] - 1))
                if abs(rr - target_rr) < 0.01: break
                if rr < target_rr:
                    eps_min = eps_mid
                else:
                    eps_max = eps_mid

            os.environ['PYOPENCL_CTX'] = '0'
            rp_gen = RecurrencePlotGenerator(signal=signal, time_delay=tau, embedding_dim=m,path=output_dir, epsilon=eps_mid, unthresholded=False)
            res, _ = rp_gen.compute_rqa_binary()
            return [res.recurrence_rate, res.determinism, res.laminarity, res.trapping_time, res.entropy_diagonal_lines]
        except:
            return [0.0] * len(metrics_list)

    def extract_features(self, pipeline_name):
        with h5py.File(self.DECOMPOSED_PATH, 'r') as f:
            n_samples = f['signals'].shape[0]
            data = {band: f[pipeline_name][band][:] for band in self.BAND_NAMES}

        params = {'max_lag': self.MAX_LAG_AMI, 'max_dim': self.MAX_DIM_FNN, 'fnn_thresh': self.FNN_THRESHOLD}

        def process_sample(i):
            sample_feats = []
            for band in self.BAND_NAMES:
                sample_feats.extend(
                    self._compute_rqa_worker(data[band][i], params, self.TARGET_RR, self.RQA_METRICS, self.OUTPUT_DIR))
            return sample_feats

        results = Parallel(n_jobs=self.n_jobs)(delayed(process_sample)(i) for i in tqdm(range(n_samples)))
        return np.array(results)

    def extract_features_single_scale(self):
        """Extract RQA features from raw signals without wavelet decomposition (baseline)."""
        with h5py.File(self.DECOMPOSED_PATH, 'r') as f:
            signals = f['signals'][:]
            n_samples = signals.shape[0]

        params = {'max_lag': self.MAX_LAG_AMI, 'max_dim': self.MAX_DIM_FNN, 'fnn_thresh': self.FNN_THRESHOLD}

        def process_sample(i):
            return self._compute_rqa_worker(signals[i], params, self.TARGET_RR, self.RQA_METRICS, self.OUTPUT_DIR)

        results = Parallel(n_jobs=self.n_jobs)(delayed(process_sample)(i) for i in tqdm(range(n_samples)))
        return np.array(results)

    def calculate_fisher_score(self, X, y):
        n_samples, n_features = X.shape
        classes = np.unique(y)
        global_mean = np.mean(X, axis=0)
        numerator, denominator = np.zeros(n_features), np.zeros(n_features)

        for c in classes:
            X_c = X[y == c]
            n_c = X_c.shape[0]
            class_mean = np.mean(X_c, axis=0)
            class_var = np.var(X_c, axis=0)
            numerator += n_c * (class_mean - global_mean) ** 2
            denominator += n_c * class_var

        return numerator / (denominator + 1e-12)

    def compare_pipelines_fisher(self, features_dict, y):
        print(f"\n{'Pipeline':<15} | {'Mean Fisher Score':<18} | {'Max Fisher Score':<18}")
        print("-" * 55)
        rankings = {}
        for name, X in features_dict.items():
            X_clean = np.nan_to_num(X)
            scores = self.calculate_fisher_score(X_clean, y)
            mean_fs, max_fs = np.mean(scores), np.max(scores)
            rankings[name] = mean_fs
            print(f"{self.PIPELINE_LABELS[name]:<15} | {mean_fs:<18.4f} | {max_fs:<18.4f}")
        return rankings

    def visualize_fisher_heatmap(self, exp_obj, pipeline_name, features_matrix, y):
        X_clean = np.nan_to_num(features_matrix)
        scores = exp_obj.calculate_fisher_score(X_clean, y)

        # Handle single-scale baseline (5 features) differently from multi-scale (25 features)
        if pipeline_name == 'single_scale':
            # Single-scale: 1 row (no bands), 5 columns (metrics)
            df_plot = pd.DataFrame(
                [scores],
                index=['Single-Scale'],
                columns=exp_obj.RQA_METRICS
            )
            plt.figure(figsize=(10, 3))
        else:
            # Multi-scale: 5 rows (bands), 5 columns (metrics)
            num_bands = len(exp_obj.BAND_NAMES)
            num_metrics = len(exp_obj.RQA_METRICS)
            score_matrix = scores.reshape(num_bands, num_metrics)

            df_plot = pd.DataFrame(
                score_matrix,
                index=exp_obj.BAND_NAMES,
                columns=exp_obj.RQA_METRICS
            )
            plt.figure(figsize=(10, 6))

        sns.heatmap(df_plot, annot=True, cmap='YlGnBu', fmt=".2f", cbar_kws={'label': 'Fisher Score'})
        plt.title(f'Fisher Discriminability Map: {exp_obj.PIPELINE_LABELS[pipeline_name]}')

        if pipeline_name == 'single_scale':
            plt.ylabel('Analysis Type')
        else:
            plt.ylabel('Wavelet Sub-band (Scale)')

        plt.xlabel('RQA Metric')
        plt.tight_layout()
        plt.show()

    def compare_metrics(self, features_dict, y):
        print(f"\n{'Pipeline':<15} | {'Mean Fisher':<12} | {'Silhouette':<12}")
        print("-" * 45)

        results = {}
        for name, X in features_dict.items():
            X_clean = np.nan_to_num(X)

            f_scores = self.calculate_fisher_score(X_clean, y)
            m_fisher = np.mean(f_scores)

            try:
                m_sil = silhouette_score(X_clean, y)
            except:
                m_sil = 0.0

            results[name] = {'fisher': m_fisher, 'silhouette': m_sil}

        return results

    def visualize_multi_scale_decomposition(self):
        np.random.seed(42)
        s_orig = {'Sine': self.synthetic_data_generator.generate_sine(), 'Lorenz': self.synthetic_data_generator.generate_lorenz_signal(), 'Noise': np.random.randn(self.SIGNAL_LENGTH)}
        class_names = list(s_orig.keys())
        colors = {'Sine': '#1f77b4', 'Lorenz': '#ff7f0e', 'Noise': '#2ca02c'}

        # 2. Loop pipelines
        pipeline_obj = Pipeline(pipeline_name='all', level=self.LEVEL, wavelet=self.WAVELET)

        for p_name in self.PIPELINES:
            print(f"  Visualizing {self.PIPELINE_LABELS[p_name]}...")

            fig = plt.figure(figsize=(18, 12), constrained_layout=True)
            fig.suptitle(f'Multi-Scale Decomposition Profile: {self.PIPELINE_LABELS[p_name]}', fontsize=16,
                         fontweight='bold')

            # Setup GridSpec for 3 classes
            gs_main = gridspec.GridSpec(3, 1, figure=fig, hspace=0.3)
            func = getattr(pipeline_obj, f"pipeline_{p_name}")

            for c_idx, cls in enumerate(class_names):
                # Class GridSpec: (1x2) - Left (Original), Right (Bands Stack)
                gs_class = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_main[c_idx], width_ratios=[1, 4],
                                                            wspace=0.05)
                signal = s_orig[cls]

                # A. Original Plot
                ax_orig = fig.add_subplot(gs_class[0])
                ax_orig.plot(signal, color=colors[cls], lw=1.0)
                ax_orig.set_title(f'{cls} (Original)', fontweight='bold')
                ax_orig.set_ylabel('Amplitude')
                ax_orig.set_xlabel('Time (samples)')
                ax_orig.grid(alpha=0.3)

                # B. Subbands Decomposition (Logic for Honoring Lengths)
                subbands = func(signal)
                gs_bands = gridspec.GridSpecFromSubplotSpec(5, 1, subplot_spec=gs_class[1], hspace=0.1)

                bands_axes = []
                for b_idx, band in enumerate(self.BAND_NAMES):

                    # Align time axes only if not decimated (DWT direct)
                    if b_idx == 0:
                        if p_name != 'dwt_direct':  # Align with original
                            ax_band = fig.add_subplot(gs_bands[0], sharex=ax_orig)
                        else:  # Start independent aligned stack for short direct signals
                            ax_band = fig.add_subplot(gs_bands[0])
                    else:  # Align with previous band in stack
                        ax_band = fig.add_subplot(gs_bands[b_idx], sharex=bands_axes[0])

                    ax_band.plot(subbands[band], color=colors[cls], lw=0.8, alpha=0.8)
                    ax_band.set_ylabel(band)
                    ax_band.grid(alpha=0.3)

                    # Remove x-ticks for all but bottom band
                    if b_idx < 4:
                        ax_band.set_xticks([])
                    else:
                        ax_band.set_xlabel('Time (samples)')

                    bands_axes.append(ax_band)

            filename = f"msra_profile_{p_name}.png"
            plt.savefig(os.path.join(self.OUTPUT_DIR, filename), dpi=300)
            print(f"    Saved visualization: {filename}")
            plt.show()

    def visualize_pca_clusters(self, features_dict, y):
        classes = {0: 'Sine (Order)', 1: 'Lorenz (Chaos)', 2: 'Noise (Stochastic)'}
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

        fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
        fig.suptitle('Feature Space Clustering: Multi-Scale RQA (PCA Projection)', fontsize=18, fontweight='bold')

        axes_flat = axes.flatten()

        for i, p_name in enumerate(self.PIPELINES):
            ax = axes_flat[i]
            X = features_dict[p_name]

            # 1. Handle NaNs and Infinite values
            X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            # 2. Check for zero variance (DWT Direct case)
            if np.all(X_clean == 0) or X_clean.std() == 0:
                ax.text(0.5, 0.5, 'DATA COLLAPSE\n(Zero Variance)',
                        ha='center', va='center', fontsize=14, color='red', fontweight='bold')
                ax.set_title(f'{self.PIPELINE_LABELS[p_name]} (FAILED)', color='red')
                continue

            # 3. Standardize and Run PCA
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_clean)

            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            var_explained = np.sum(pca.explained_variance_ratio_) * 100

            # 4. Plot individual classes
            for class_id, label in classes.items():
                mask = (y == class_id)
                ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                           label=label, color=colors[class_id], alpha=0.7, edgecolors='w', s=60)

            ax.set_title(f'{self.PIPELINE_LABELS[p_name]} (Var: {var_explained:.1f}%)', fontweight='bold')
            ax.set_xlabel('Principal Component 1')
            ax.set_ylabel('Principal Component 2')
            ax.legend(loc='best', fontsize='small')
            ax.grid(alpha=0.2)

        plt.savefig(os.path.join(self.OUTPUT_DIR, 'pca_clusters_comparison.png'), dpi=300)
        plt.show()


    def perform_feature_anova(self, X, y):
        """
        Performs a One-Way ANOVA for each feature to calculate p-values.

        Returns:
            pd.DataFrame: A table containing F-statistics and p-values for each feature.
        """
        n_features = X.shape[1]
        classes = np.unique(y)

        results = []

        # We iterate through each feature (column)
        for i in range(n_features):
            # Group the data for this specific feature by class
            groups = [X[y == c, i] for c in classes]

            # Perform One-Way ANOVA
            # H0: The means of all classes are equal
            # H1: At least one class mean is different
            f_stat, p_val = stats.f_oneway(*groups)
            stat, pval = stats.kruskal(*groups)

            results.append({
                'Feature_Index': i,
                'F_Statistic': f_stat,
                'p_value': p_val,
                'Significant_05': p_val < 0.05,
                'Significant_01': p_val < 0.01,
                'kruskal_statistic': stat,
                'kruskal_statistic_pvalue': pval,
                'Significant_05_kr': pval < 0.05,
                'Significant_01_kr': pval < 0.01,
            })

        return pd.DataFrame(results)


if __name__ == '__main__':
    exp = ParallelExperiment(n_jobs=-1)

    X, y = exp.synthetic_data_generator.generate_dataset()
    exp.decompose_and_save(X, y)

    features_single_scale = exp.extract_features_single_scale()
    features_swt_direct = exp.extract_features('swt_direct')
    features_dwt_direct = exp.extract_features('dwt_direct')
    features_swt_recon = exp.extract_features('swt_recon')
    features_dwt_recon = exp.extract_features('dwt_recon')

    pipelines_to_test = {
        'single_scale': features_single_scale,
        'swt_direct': features_swt_direct,
        'swt_recon': features_swt_recon,
        'dwt_direct': features_dwt_direct,
        'dwt_recon': features_dwt_recon,
    }

    # for name, feat_matrix in pipelines_to_test.items():
    #     exp.visualize_fisher_heatmap(exp, name, feat_matrix, y)
    #
    # anova_single_scale = exp.perform_feature_anova(features_single_scale, y)
    #
    # anova_table_swt_direct = exp.perform_feature_anova(features_swt_direct, y)
    #
    # anova_table_swt_recon = exp.perform_feature_anova(features_swt_recon, y)
    #
    # anova_table_dwt_direct = exp.perform_feature_anova(features_dwt_direct, y)
    #
    # anova_table_dwt_recon = exp.perform_feature_anova(features_dwt_recon, y)
    #
    # print("\n" + "="*60)
    # print("Single-Scale (Baseline)")
    # print("="*60)
    # print(anova_single_scale.sort_values(by='F_Statistic', ascending=False).head(10))
    # print("\n" + "="*60)
    # print("SWT Direct")
    # print("="*60)
    # print(anova_table_swt_direct.sort_values(by='F_Statistic', ascending=False).head(10))
    # print("\n" + "="*60)
    # print("SWT Recon")
    # print("="*60)
    # print(anova_table_swt_recon.sort_values(by='F_Statistic', ascending=False).head(10))
    # print("\n" + "="*60)
    # print("DWT Direct")
    # print("="*60)
    # print(anova_table_dwt_direct.sort_values(by='F_Statistic', ascending=False).head(10))
    # print("\n" + "="*60)
    # print("DWT Recon")
    # print("="*60)
    # print(anova_table_dwt_recon.sort_values(by='F_Statistic', ascending=False).head(10))
    #
    # # Compare metrics across all pipelines
    # print("\n" + "="*60)
    # print("OVERALL COMPARISON: Fisher Score & Silhouette")
    # print("="*60)
    # results = exp.compare_metrics(pipelines_to_test, y)
    # for pipeline_name, metrics in results.items():
    #     print(f"{exp.PIPELINE_LABELS[pipeline_name]:<30} | Fisher: {metrics['fisher']:.4f} | Silhouette: {metrics['silhouette']:.4f}")

    # exp.visualize_pca_clusters(pipelines_to_test, y)
    exp.visualize_multi_scale_decomposition()