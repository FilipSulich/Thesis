import numpy as np
from scipy.stats import skew, kurtosis
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray

class URPFeatures:
    """extracts texture and statistical features from URP images for classification"""

    @staticmethod
    def extract(URP_rgb):
        """
        extracts 21 texture and statistical features from RGB URP image

        Args:
            URP_rgb (np.array): RGB URP image of shape (H, W, 3) with values in range [0,1]

        Returns:
            np.array: feature vector of 21 elements containing per-channel statistics (12), entropy (1), and GLCM texture features (8)
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
