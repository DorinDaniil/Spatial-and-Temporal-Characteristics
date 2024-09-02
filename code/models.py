import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests
from skimage.filters import threshold_otsu
from skimage.morphology import binary_closing, binary_opening, ball
from pyriemann.tangentspace import TangentSpace
from pyriemann.estimation import Shrinkage
from sklearn.pipeline import make_pipeline
from pyriemann.estimation import Covariances
from typing import Tuple
import numpy as np


def ExtractSegments(X: np.ndarray, s: np.ndarray, tau: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract segments from array X based on classes in array s.

    Parameters
    ----------
    X : np.ndarray
        Array from which segments will be extracted.
    s : np.ndarray
        Array of classes defining the segments.
    tau : int
        Length of the segments.

    Returns
    -------
    X_segments : np.ndarray
        Array of segments from X.
    s_segments : np.ndarray
        Array of class segments from s.
    y : np.ndarray
        Array of classes for each segment.
    """

    # Initialize lists to store the segments
    X_segments: list[np.ndarray] = []
    s_segments: list[np.ndarray] = []
    y: list[int] = []

    # Loop over all possible classes
    for k in np.unique(s):
        if k == 0:
            continue

        # Find the start and end indices of each segment of class k
        b = np.where(np.diff(s) == k)[0] + 1
        e = np.where(np.diff(s) == -k)[0]

        # Loop over each segment of class k
        for j in range(len(b)):
            # Calculate the length of the current segment
            tau_j = e[j] - b[j] + 1

            # Calculate delta1 and delta2
            delta1 = int(np.floor((tau - tau_j) / 2))
            delta2 = int(np.ceil((tau - tau_j) / 2))

            # Extract the segment from X and s
            X_j = X[b[j] - delta1:e[j] + delta2 + 1]
            s_j = s[b[j] - delta1:e[j] + delta2 + 1]

            # Add the segment to the lists
            X_segments.append(X_j)
            s_segments.append(s_j)
            y.append(k)

    # Convert the lists to numpy arrays
    X_segments = np.array(X_segments, dtype=object)
    s_segments = np.array(s_segments, dtype=object)
    y = np.array(y)

    return X_segments, s_segments, y

def ccw(
    X: np.ndarray, 
    s: np.ndarray, 
    mu: float, 
    k_s: int, 
    Delta_t: float, 
    h: int, 
    masked_brain: bool = False,
    ablation: bool = False
    ) -> tuple:
    """
    Method Cross-Correlation Weighting.

    Args:
    X (np.ndarray): fMRI time series.
    s (np.ndarray): Stimulus time series.
    mu (float): Sampling rate.
    k_s (int): Kernel size for pooling.
    Delta_t (float): BOLD signal delay.
    h (int): Number of most significant pixels.
    masked_brain (bool): Flag to use brain mask.

    Returns:
    tuple: A tuple of two elements: mask of most significant pixels, 
           p-values after Holm's correction.
    """

    X = torch.tensor(X).clone()  # (tau, X, Y, Z)
    s = torch.tensor(s).clone()  # (tau,)

    if masked_brain:
        mean_X = torch.mean(X, axis=0)
        mean_X = (mean_X - mean_X.min()) / (mean_X.max() - mean_X.min())
        threshold = threshold_otsu(mean_X.numpy())
        binary_mask = mean_X.numpy() > threshold
        brain_mask = binary_closing(binary_mask, ball(2))
        brain_mask = binary_opening(brain_mask, ball(2)).astype(np.int8)

    # 3D Average Pooling
    X_prime = F.avg_pool3d(X, kernel_size=k_s)  # (tau, X/k_s, Y/k_s, Z/k_s)

    if masked_brain:
        mean_X_prime = torch.mean(X_prime, axis=0)
        mean_X_prime = (mean_X_prime - mean_X_prime.min()) / (mean_X_prime.max() - mean_X_prime.min())
        threshold = threshold_otsu(mean_X_prime.numpy())
        binary_mask = mean_X_prime.numpy() > threshold
        brain_mask_pooled = binary_closing(binary_mask, ball(2))
        brain_mask_pooled = binary_opening(brain_mask_pooled, ball(2)).astype(np.int8)

    # Z-normalization of time series
    X_prime_hat = ((X_prime - X_prime.mean(dim=0, keepdims=True)) / (X_prime.std(dim=0) + 1e-9)).float()
    s_hat = ((s.float() - s.float().mean()) / s.float().std()).float()

    # Cross-correlation computation
    tau = X_prime_hat.shape[0]
    C = torch.zeros((tau-1, X_prime_hat.shape[1], X_prime_hat.shape[2], X_prime_hat.shape[3]))
    for p in range(tau-1):
        C[p] = torch.einsum('i,itxy->txy', s_hat[:tau-p], X_prime_hat[p:]) / (tau-1)

    # BOLD Delay and mask computation
    p_BOLD = int(np.floor(Delta_t * mu))
    C_p_BOLD = C[p_BOLD]

    if masked_brain:
        C_p_BOLD = C_p_BOLD * torch.tensor(brain_mask_pooled)

    _, top_h_indices = torch.topk(C_p_BOLD.view(-1), h)
    M_c = torch.zeros_like(C_p_BOLD)
    M_c.view(-1)[top_h_indices] = 1

    # Perform statistical test and Holm's method
    p_values = np.zeros((X_prime_hat.shape[1], X_prime_hat.shape[2], X_prime_hat.shape[3]))
    for i in range(X_prime_hat.shape[1]):
        for j in range(X_prime_hat.shape[2]):
            for k in range(X_prime_hat.shape[3]):
                _, p_value = pearsonr(s_hat.numpy(), X_prime_hat[:, i, j, k].numpy(), alternative='greater')
                p_values[i, j, k] = p_value

    p_values_flattened = p_values.flatten()
    _, corrected_p_values, _, _ = multipletests(p_values_flattened, method='holm')
    corrected_p_values = corrected_p_values.reshape(p_values.shape)
    p_values = corrected_p_values

    # Upsample
    p_values = F.upsample(torch.tensor(p_values).unsqueeze(0).unsqueeze(0), size=(X.shape[1], X.shape[2], X.shape[3]), mode='nearest')
    p_values = p_values.squeeze(0).squeeze(0)

    M = F.upsample(M_c.unsqueeze(0).unsqueeze(0), size=(X.shape[1], X.shape[2], X.shape[3]), mode='nearest')
    M = M.squeeze(0).squeeze(0)

    if masked_brain:
        M = M * torch.tensor(brain_mask)
    if ablation:
        return M_c
    return M.numpy(), p_values.numpy()


class Encoder:
    def __init__(
        self, 
        mu: float, 
        k_s: int, 
        Delta_t: float, 
        h: int, 
        stat_test: bool = False
    ) -> None:
        """
        Encoder class.

        Args:
        mu (float): Sampling rate.
        k_s (int): Kernel size for pooling.
        Delta_t (float): BOLD signal delay.
        h (int): Number of most significant pixels.
        stat_test (bool): Flag for statistical test.
        """
        self.mu = mu
        self.k_s = k_s
        self.Delta_t = Delta_t
        self.h = h
        self.stat_test = stat_test
        self.masks = None
        self.preps = None

    def _avg_pool3d(self, X: np.ndarray, k_s: int) -> torch.Tensor:
        """
        Performs 3D average pooling.

        Args:
        X (np.ndarray): Input data.
        k_s (int): Kernel size for pooling.

        Returns:
        torch.Tensor: Pooled data.
        """
        X = torch.tensor(X).clone()
        return F.avg_pool3d(X, kernel_size=k_s)

    def _ccw(
        self, 
        X: np.ndarray, 
        s: np.ndarray, 
        mu: float, 
        k_s: int, 
        Delta_t: float, 
        h: int, 
        stat_test: bool = True
    ) -> np.ndarray:
        """
        Performs correlation analysis.

        Args:
        X (np.ndarray): Input data.
        s (np.ndarray): Stimulus time series.
        mu (float): Sampling rate.
        k_s (int): Kernel size for pooling.
        Delta_t (float): BOLD signal delay.
        h (int): Number of most significant pixels.
        stat_test (bool): Flag for statistical test.

        Returns:
        np.ndarray: Mask.
        """
        # 1. Input: temporal fMRI series X with sampling rate mu and stimulus time series s
        X = torch.tensor(X).clone()  # (tau, X, Y, Z)
        s = torch.tensor(s).clone()  # (tau,)

        # 2. 3D Average Pooling
        X_prime = self._avg_pool3d(X, k_s)  # (tau, X/k_s, Y/k_s, Z/k_s)

        # 3. Z-normalization of time series
        X_prime_hat = ((X_prime - X_prime.mean(dim=0, keepdims=True)) / (X_prime.std(dim=0) + 1e-9)).float()
        s_hat = ((s.float() - s.float().mean()) / s.float().std()).float()

        # 4. Cross-correlation computation
        tau = X_prime_hat.shape[0]
        C = torch.zeros((tau-1, X_prime_hat.shape[1], X_prime_hat.shape[2], X_prime_hat.shape[3]))
        for p in range(tau-1):
            C[p] = torch.einsum('i,itxy->txy', s_hat[:tau-p], X_prime_hat[p:]) / (tau-1)

        # 5. BOLD Delay and mask computation
        p_BOLD = int(np.floor(Delta_t * mu))
        C_p_BOLD = C[p_BOLD]

        _, top_h_indices = torch.topk(C_p_BOLD.view(-1), h)
        M_c = torch.zeros_like(C_p_BOLD)
        M_c.view(-1)[top_h_indices] = 1
        if not stat_test:
            return M_c.numpy()
        
        # 6. Perform statistical test and Holm's method
        if stat_test:
            p_values = np.zeros((X_prime_hat.shape[1], X_prime_hat.shape[2], X_prime_hat.shape[3]))
            for i in range(X_prime_hat.shape[1]):
                for j in range(X_prime_hat.shape[2]):
                    for k in range(X_prime_hat.shape[3]):
                        # Perform statistical test
                        _, p_value = pearsonr(s_hat.numpy(), X_prime_hat[:, i, j, k].numpy(), alternative='greater')
                        p_values[i, j, k] = p_value

            # Apply Holm's method for multiple comparisons
            p_values_flattened = p_values.flatten()
            _, corrected_p_values, _, _ = multipletests(p_values_flattened, method='holm')

            # Reshape and replace original p_values
            corrected_p_values = corrected_p_values.reshape(p_values.shape)
            p_values = corrected_p_values

            M = (p_values<0.05).astype(int)
        return M

    def fit_masks(
                self, 
                X: np.ndarray, 
                s: np.ndarray, 
                y: np.ndarray
    ) -> None:
        """
        Fits masks for each class.

        Args:
        X (np.ndarray): Input data.
        s (np.ndarray): Stimulus time series.
        y (np.ndarray): Class labels.
        """
        self.masks = {}
        for cls in np.unique(y):
            X_cls = X[y==cls].copy()
            s_cls = s[y==cls].copy()
            bin_s = s_cls.copy()
            bin_s[bin_s!=0]=1
            bin_s = np.concatenate(bin_s, axis=0)
            X_cls = np.concatenate(X_cls, axis=0)
            delta_s = np.diff(bin_s)
            delta_X = np.diff(X_cls, axis=0)
            self.masks[cls] = self._ccw(delta_X, delta_s, self.mu, self.k_s, self.Delta_t, self.h, self.stat_test)


    def _apply_cls_mask(
        self, 
        X: np.ndarray, 
        cls: int
    ) -> np.ndarray:
        """
        Applies the mask for a specific class to the input data.

        Args:
        X (np.ndarray): Input data.
        cls (int): Class label.

        Returns:
        np.ndarray: Masked data.
        """
        X = (self._avg_pool3d(X, k_s=self.k_s)).numpy()
        # Expand the mask dimensions to match the fMRI data
        mask = np.expand_dims(self.masks[cls], axis=(0, 1))
        mask = np.tile(mask, (X.shape[0], X.shape[1], 1, 1, 1))

        # Apply the mask to the fMRI data
        masked_X = X * mask
        # Flatten the masked data
        masked_X = masked_X.reshape((masked_X.shape[0], masked_X.shape[1], -1))

        # Remove voxels with zero values
        nonzero_voxels = np.where(self.masks[cls].flatten() > 0)
        masked_X = masked_X[:, :, nonzero_voxels]

        masked_X = np.squeeze(masked_X) 
        masked_X = np.transpose(masked_X, (0, 2, 1))
        return masked_X


    def _binarize_rel_labels(
        self, 
        y: np.ndarray, 
        cls: int
    ) -> np.ndarray:
        """
        Binarizes the relative labels.

        Args:
        y (np.ndarray): Class labels.
        cls (int): Class label.

        Returns:
        np.ndarray: Binarized labels.
        """
        bin_y = np.zeros_like(y)
        bin_y[np.where(y==cls)]=1
        return bin_y


    def _fit_tsm(
        self, 
        masked_X: np.ndarray, 
        bin_y: np.ndarray
    ) -> object:
        """
        Fits the tangent space model.

        Args:
        masked_X (np.ndarray): Masked data.
        bin_y (np.ndarray): Binarized labels.

        Returns:
        object: Fitted model.
        """
        covest = Covariances()
        reg = Shrinkage(shrinkage=1e-3)
        ts = TangentSpace()
        preprocess = make_pipeline(covest,reg,ts)
        preprocess.fit(masked_X, bin_y)
        return preprocess


    def fit(
        self, 
        X: np.ndarray, 
        s: np.ndarray, 
        y: np.ndarray
    ) -> None:
        """
        Fits the model.

        Args:
        X (np.ndarray): Input data.
        s (np.ndarray): Stimulus time series.
        y (np.ndarray): Class labels.
        """
        self.fit_masks(X, s, y)
        self.preps = {}
        self.classes = np.unique(y)
        for cls in np.unique(y):
            masked_X = self._apply_cls_mask(X, cls)
            bin_y = self._binarize_rel_labels(y, cls)
            prep = self._fit_tsm(masked_X, bin_y)
            self.preps[cls] = prep


    def transform_cls(
        self, 
        X: np.ndarray, 
        cls: int
    ) -> np.ndarray:
        """
        Transforms the data for a specific class.

        Args:
        X (np.ndarray): Input data.
        cls (int): Class label.

        Returns:
        np.ndarray: Transformed data.
        """
        masked_X = self._apply_cls_mask(X, cls)
        prep_X = self.preps[cls].transform(masked_X)
        return prep_X
    
    def transform(
        self, 
        X: np.ndarray
    ) -> np.ndarray:
        """
        Transforms the input data into embedding vectors.

        Args:
        X (np.ndarray): Input data.

        Returns:
        np.ndarray: Transformed data.
        """
        # Finally, get the embedding vectors
        new_X = []
        for cls in self.classes:
            masked_X = self._apply_cls_mask(X, cls)
            prep_X = self.preps[cls].transform(masked_X)
            new_X.append(prep_X)
        new_X = np.concatenate(new_X, axis=1)
        return new_X