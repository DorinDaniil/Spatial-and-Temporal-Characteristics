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
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from nilearn.image import load_img
from dataloader import HaxbyDataset


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



def get_results_for_sub(num, stat_test=True):
    dataset = HaxbyDataset()
    fmris, labels = dataset.get_sub_data(num)

    class_dict = {'rest': 0, 'bottle': 1, 'cat': 2, 'chair': 3, 'face': 4, 'house': 5, 'scissors': 6, 'scrambledpix': 7, 'shoe': 8}
    stimuli = np.array([class_dict[label] for label in labels['labels'].values])
    fmris_transposed = fmris.transpose((3, 0, 1, 2))

    X, s, y = ExtractSegments(fmris_transposed, stimuli, tau=19)
    y=y-1

    X = X.astype('float64')
    s = s.astype('int')
    X_train, X_test, s_train, _, y_train, y_test = train_test_split(X, s, y, test_size=0.20, random_state=42, stratify=y)
    clf = Encoder(mu=2.5, k_s=4, Delta_t=0.0, h=10, stat_test=stat_test)
    clf.fit(X_train, s_train, y_train)

    X_ptrain = clf.transform(X_train)
    X_ptest = clf.transform(X_test)

    # 1. Logistic Regression
    print("Logistic Regression")
    lr = LogisticRegression(C=1e2, class_weight='balanced', random_state=42)
    lr.fit(X_ptrain, y_train)
    lr_pred = lr.predict(X_ptest)
    macro_f1_logreg = f1_score(y_test, lr_pred, average='macro')
    micro_f1_logreg = f1_score(y_test, lr_pred, average='micro')
    acc_logreg = accuracy_score(y_test, lr_pred)
    print(f"Macro-average F1-Score: {macro_f1_logreg:.4f}")
    print(f"Micro-average F1-Score: {micro_f1_logreg:.4f}")
    print(f"Accuracy: {acc_logreg:.4f}\n")

    # 2. MLP
    print("MLP")
    clf = MLPClassifier(hidden_layer_sizes=(128, 128), max_iter=200, activation = 'logistic')
    clf.fit(X_ptrain, y_train)

    y_pred = clf.predict(X_ptest)
    macro_f1_mlp = f1_score(y_test, y_pred, average='macro')
    micro_f1_mlp = f1_score(y_test, y_pred, average='macro')
    acc_mlp = accuracy_score(y_test, y_pred)
    print(f"Macro-average F1-Score: {macro_f1_mlp:.4f}")
    print(f"Micro-average F1-Score: {micro_f1_mlp:.4f}")
    print(f"Accuracy: {acc_mlp:.4f}")

    return macro_f1_logreg, micro_f1_logreg, acc_logreg, macro_f1_mlp, micro_f1_mlp, acc_mlp


# Define the LSTM model
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Define the Attention-based model
class AttentionClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, attention_dim, output_dim):
        super(AttentionClassifier, self).__init__()
        self.hidden_dim = hidden_dim

        # Linear transformation for input
        self.linear_in = nn.Linear(input_dim, hidden_dim)

        # Attention layer
        self.attention = nn.Linear(hidden_dim, attention_dim)
        self.attention_combine = nn.Linear(attention_dim, 1)

        # Output layer
        self.fc_output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Linear transformation of input
        x = self.linear_in(x)

        # Apply attention mechanism
        attn_weights = torch.softmax(self.attention_combine(torch.tanh(self.attention(x))), dim=1)
        context = torch.sum(attn_weights * x, dim=1)

        # Output layer
        out = self.fc_output(context)

        return out

# Prepare the data
def prepare_data(fmris, labels, mask):
    class_dict = {'rest': 0, 'bottle': 1, 'cat': 2, 'chair': 3, 'face': 4, 'house': 5, 'scissors': 6, 'scrambledpix': 7, 'shoe': 8}
    stimuli = np.array([class_dict[label] for label in labels['labels'].values])
    fmris_transposed = fmris.transpose((3, 0, 1, 2))

    # Segment the time series by stimuli classes
    X, s, y = ExtractSegments(fmris_transposed, stimuli, tau=19)
    y = y - 1

    # Apply the mask to reduce dimensionality
    masked_X = []
    for segment in X:
        segment_masked = segment * mask
        segment_masked = segment_masked.reshape((segment_masked.shape[0], -1))
        nonzero_voxels = np.where(mask.flatten() > 0)[0]
        segment_masked = segment_masked[:, nonzero_voxels]
        masked_X.append(segment_masked)

    masked_X = np.array(masked_X)

    return masked_X, y

# Function to train LSTM model
def train_lstm_model(subject_num, lr=1e-5, num_epochs=1):
    # Load the data
    class_dict = {'rest': 0, 'bottle': 1, 'cat': 2, 'chair': 3, 'face': 4, 'house': 5, 'scissors': 6, 'scrambledpix': 7, 'shoe': 8}
    dataset = HaxbyDataset()
    fmris, labels = dataset.get_sub_data(subject_num)
    mask = load_img(dataset.data_files.mask_vt[subject_num-1]).get_fdata()

    # Prepare the data
    X, y = prepare_data(fmris, labels, mask)

    # Ensure the data is numeric and properly shaped
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    # Reshape the data to (batch_size, sequence_length, input_dim)
    X = X.reshape((X.shape[0], X.shape[1], -1))

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Convert data to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # Define the model, loss function, and optimizer
    input_dim = X_train.shape[2]
    hidden_dim = 128
    layer_dim = 2
    output_dim = len(class_dict)

    model = LSTMClassifier(input_dim, hidden_dim, layer_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train the model
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        # Calculate training metrics
        _, predicted_train = torch.max(outputs, 1)
        train_accuracy = accuracy_score(y_train, predicted_train)
        train_macro_f1 = f1_score(y_train, predicted_train, average='macro')
        train_micro_f1 = f1_score(y_train, predicted_train, average='micro')

        if (epoch+1) % 1 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, '
                  f'Train Accuracy: {train_accuracy:.4f}, Train Macro F1: {train_macro_f1:.4f}, '
                  f'Train Micro F1: {train_micro_f1:.4f}')

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)
        macro_f1 = f1_score(y_test, predicted, average='macro')
        micro_f1 = f1_score(y_test, predicted, average='micro')
        accuracy = accuracy_score(y_test, predicted)
        print(f'Test Macro-average F1-Score: {macro_f1:.4f}')
        print(f'Test Micro-average F1-Score: {micro_f1:.4f}')
        print(f'Test Accuracy: {accuracy:.4f}')

# Function to train Attention model
def train_attention_model(subject_num, lr=1e-5, num_epochs=1):
    # Load the data
    class_dict = {'rest': 0, 'bottle': 1, 'cat': 2, 'chair': 3, 'face': 4, 'house': 5, 'scissors': 6, 'scrambledpix': 7, 'shoe': 8}
    dataset = HaxbyDataset()
    fmris, labels = dataset.get_sub_data(subject_num)
    mask = load_img(dataset.data_files.mask_vt[subject_num-1]).get_fdata()

    # Prepare the data
    X, y = prepare_data(fmris, labels, mask)

    # Ensure the data is numeric and properly shaped
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    # Reshape the data to (batch_size, sequence_length, input_dim)
    X = X.reshape((X.shape[0], X.shape[1], -1))

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Convert data to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # Define the model, loss function, and optimizer
    input_dim = X_train.shape[2]
    hidden_dim = 128
    attention_dim = 64  # Dimension of the attention layer
    output_dim = len(class_dict)

    model = AttentionClassifier(input_dim, hidden_dim, attention_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train the model
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        # Calculate training metrics
        _, predicted_train = torch.max(outputs, 1)
        train_accuracy = accuracy_score(y_train, predicted_train)
        train_macro_f1 = f1_score(y_train, predicted_train, average='macro')
        train_micro_f1 = f1_score(y_train, predicted_train, average='micro')

        if (epoch+1) % 1 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, '
                  f'Train Accuracy: {train_accuracy:.4f}, Train Macro F1: {train_macro_f1:.4f}, '
                  f'Train Micro F1: {train_micro_f1:.4f}')

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)
        macro_f1 = f1_score(y_test, predicted, average='macro')
        micro_f1 = f1_score(y_test, predicted, average='micro')
        accuracy = accuracy_score(y_test, predicted)
        print(f'Test Macro-average F1-Score: {macro_f1:.4f}')
        print(f'Test Micro-average F1-Score: {micro_f1:.4f}')
        print(f'Test Accuracy: {accuracy:.4f}')