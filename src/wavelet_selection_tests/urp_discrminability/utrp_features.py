import numpy as np
from scipy.stats import skew, kurtosis
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray

class URPFeatures:
    """Extract texture statistics from URP images"""

    @staticmethod
    def extract(URP_rgb):
        """
        Pull features from URP image (H, W, 3) in range [0,1].

        Gets:
        - Per-channel stats: mean, std, skew, kurtosis (12 total)
        - Grayscale entropy (1)
        - GLCM texture at different angles/distances (8)

        Total: 21 features
        """
        features = []

        # Per-channel stats (R, G, B)
        for c in range(3):
            ch = URP_rgb[:, :, c].ravel()
            ch_skew = skew(ch)
            ch_kurt = kurtosis(ch)

            # handle NaN from constant channels
            features.extend([
                float(ch.mean()),
                float(ch.std()),
                0.0 if not np.isfinite(ch_skew) else float(ch_skew),
                0.0 if not np.isfinite(ch_kurt) else float(ch_kurt),
            ])

        # Grayscale entropy
        gray = rgb2gray(URP_rgb)
        hist, _ = np.histogram(gray, bins=64, range=(0, 1), density=True)
        hist = hist + 1e-10  # avoid log(0)
        entropy = -np.sum(hist * np.log(hist))
        features.append(float(entropy))

        # GLCM texture on quantized gray
        gray_u8 = (gray * 255).astype(np.uint8)
        glcm = graycomatrix(
            gray_u8,
            distances=[1, 3],
            angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
            levels=256,
            symmetric=True,
            normed=True
        )

        for prop in ['contrast', 'correlation', 'energy', 'homogeneity']:
            val = graycoprops(glcm, prop)
            mean_val = val.mean()
            std_val = val.std()
            features.append(0.0 if not np.isfinite(mean_val) else float(mean_val))
            features.append(0.0 if not np.isfinite(std_val) else float(std_val))

        return np.array(features, dtype=np.float32)
