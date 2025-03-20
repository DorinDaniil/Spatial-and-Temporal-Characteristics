import numpy as np
import matplotlib.pyplot as plt
import warnings
import torch
import torch.nn.functional as F
from pyriemann.tangentspace import TangentSpace
from pyriemann.estimation import Shrinkage
from sklearn.pipeline import make_pipeline
from pyriemann.estimation import Covariances

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

from models import ExtractSegments, ccw, Encoder
from dataloader import HaxbyDataset, prepare_sub_data

def apply_cls_mask(X, M, k_s):
    # Expand the mask dimensions to match the fMRI data
    if k_s is not None:
        X = torch.tensor(X)
        X = F.avg_pool3d(X, kernel_size=k_s).numpy()
    mask = np.expand_dims(M, axis=(0, 1))
    mask = np.tile(mask, (X.shape[0], X.shape[1], 1, 1, 1))

    # Apply the mask to the fMRI data
    masked_X = X * mask
    # Flatten the masked data
    masked_X = masked_X.reshape((masked_X.shape[0], masked_X.shape[1], -1))

    # Remove voxels with zero values
    nonzero_voxels = np.where(M.flatten() > 0)
    masked_X = masked_X[:, :, nonzero_voxels]

    masked_X = np.squeeze(masked_X) 
    masked_X = np.transpose(masked_X, (0, 2, 1))
    return masked_X


# def get_ablation_1_results_for_sub(num):
#     dataset = HaxbyDataset()
#     fmris, labels = dataset.get_sub_data(num)

#     class_dict = {'rest': 0, 'bottle': 1, 'cat': 2, 'chair': 3, 'face': 4, 'house': 5, 'scissors': 6, 'scrambledpix': 7, 'shoe': 8}
#     stimuli = np.array([class_dict[label] for label in labels['labels'].values])
#     fmris_transposed = fmris.transpose((3, 0, 1, 2))

#     X, s, y = ExtractSegments(fmris_transposed, stimuli, tau=19)
#     y=y-1

#     X = X.astype('float64')
#     s = s.astype('int')
#     X_train, X_test, s_train, _, y_train, y_test = train_test_split(X, s, y, test_size=0.20, random_state=42, stratify=y)

#     s_train_new = s_train
#     s_train_new[np.where(s_train_new!=0)] = 1
#     s_train_new = np.concatenate(s_train_new)

#     X_train_new = np.concatenate(X_train, axis=0)
#     k_s = 4
#     M = ccw(X=X_train_new, s=s_train_new, mu=2.5, k_s=k_s, Delta_t=0, h=10, masked_brain=False, ablation=True)
    
#     X_train_masked = apply_cls_mask(X_train, M, k_s)
#     X_test_masked = apply_cls_mask(X_test, M, k_s)

#     X_ptest = X_test_masked.reshape((20, -1))
#     X_ptrain = X_train_masked.reshape((76, -1))

#     # 1. Logistic Regression
#     print("Logistic Regression")
#     lr = LogisticRegression(C=1e2, class_weight='balanced', random_state=42)
#     lr.fit(X_ptrain, y_train)
#     lr_pred = lr.predict(X_ptest)
#     macro_f1_logreg = f1_score(y_test, lr_pred, average='macro')
#     micro_f1_logreg = f1_score(y_test, lr_pred, average='micro')
#     acc_logreg = accuracy_score(y_test, lr_pred)
#     print(f"Macro-average F1-Score: {macro_f1_logreg:.4f}")
#     print(f"Micro-average F1-Score: {micro_f1_logreg:.4f}")
#     print(f"Accuracy: {acc_logreg:.4f}\n")

#     # 2. MLP
#     print("MLP")
#     clf = MLPClassifier(hidden_layer_sizes=(128, 128), max_iter=200, activation = 'logistic')
#     clf.fit(X_ptrain, y_train)

#     y_pred = clf.predict(X_ptest)
#     macro_f1_mlp = f1_score(y_test, y_pred, average='macro')
#     micro_f1_mlp = f1_score(y_test, y_pred, average='macro')
#     acc_mlp = accuracy_score(y_test, y_pred)
#     print(f"Macro-average F1-Score: {macro_f1_mlp:.4f}")
#     print(f"Micro-average F1-Score: {micro_f1_mlp:.4f}")
#     print(f"Accuracy: {acc_mlp:.4f}")

#     return macro_f1_logreg, micro_f1_logreg, acc_logreg, macro_f1_mlp, micro_f1_mlp, acc_mlp

def get_ablation_1_results_for_sub(num):
    dataset = HaxbyDataset()
    fmris, labels = dataset.get_sub_data(num)

    class_dict = {'rest': 0, 'bottle': 1, 'cat': 2, 'chair': 3, 'face': 4, 'house': 5, 'scissors': 6, 'scrambledpix': 7, 'shoe': 8}
    stimuli = np.array([class_dict[label] for label in labels['labels'].values])
    fmris_transposed = fmris.transpose((3, 0, 1, 2))

    X, s, y = ExtractSegments(fmris_transposed, stimuli, tau=19)
    y = y - 1

    X = X.astype('float64')
    s = s.astype('int')
    X_train, X_test, s_train, _, y_train, y_test = train_test_split(X, s, y, test_size=0.20, random_state=42, stratify=y)

    # Generate masks for each class
    masks = {}
    for cls in np.unique(y_train):
        X_cls = X_train[y_train == cls].copy()
        s_cls = s_train[y_train == cls].copy()
        bin_s = s_cls.copy()
        bin_s[bin_s != 0] = 1
        bin_s = np.concatenate(bin_s, axis=0)
        X_cls = np.concatenate(X_cls, axis=0)
        masks[cls] = ccw(X_cls, bin_s, mu=2.5, k_s=4, Delta_t=0, h=10, masked_brain=False, ablation=True)

    # Apply masks and concatenate features
    def apply_mask_and_concatenate(X, masks, k_s=4):
        masked_features = []
        for cls in masks.keys():
            mask = masks[cls]
            masked_X = X * np.expand_dims(mask, axis=(0, 1))
            masked_X = masked_X.reshape((masked_X.shape[0], masked_X.shape[1], -1))
            nonzero_voxels = np.where(mask.flatten() > 0)
            masked_X = masked_X[:, :, nonzero_voxels]
            masked_X = np.squeeze(masked_X)
            masked_X = np.transpose(masked_X, (0, 2, 1))
            masked_features.append(masked_X)
        return np.concatenate(masked_features, axis=1)

    X_train = torch.tensor(X_train)
    X_train = F.avg_pool3d(X_train, kernel_size=4).numpy()
    X_test = torch.tensor(X_test)
    X_test = F.avg_pool3d(X_test, kernel_size=4).numpy()

    X_train_masked = apply_mask_and_concatenate(X_train, masks)
    X_test_masked = apply_mask_and_concatenate(X_test, masks)

    X_ptest = X_test_masked.reshape((20, -1))
    X_ptrain = X_train_masked.reshape((76, -1))

    # Logistic Regression
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

    # MLP Classifier
    print("MLP Classifier")
    mlp = MLPClassifier(hidden_layer_sizes=(128, 128), max_iter=200, activation = 'logistic')
    mlp.fit(X_ptrain, y_train)
    mlp_pred = mlp.predict(X_ptest)
    macro_f1_mlp = f1_score(y_test, mlp_pred, average='macro')
    micro_f1_mlp = f1_score(y_test, mlp_pred, average='micro')
    acc_mlp = accuracy_score(y_test, mlp_pred)
    print(f"Macro-average F1-Score: {macro_f1_mlp:.4f}")
    print(f"Micro-average F1-Score: {micro_f1_mlp:.4f}")
    print(f"Accuracy: {acc_mlp:.4f}\n")

    return macro_f1_logreg, micro_f1_logreg, acc_logreg, macro_f1_mlp, micro_f1_mlp, acc_mlp


def get_ablation_2_results_for_sub(num):
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

    s_train_new = s_train
    s_train_new[np.where(s_train_new!=0)] = 1
    s_train_new = np.concatenate(s_train_new)

    X_train_new = np.concatenate(X_train, axis=0)
    k_s = 4
    M = ccw(X=X_train_new, s=s_train_new, mu=2.5, k_s=k_s, Delta_t=0, h=10, masked_brain=False, ablation=True)
    
    X_train_masked = apply_cls_mask(X_train, M, k_s)
    X_test_masked = apply_cls_mask(X_test, M, k_s)

    covest = Covariances()
    reg = Shrinkage(shrinkage=1e-3)
    ts = TangentSpace()
    preprocess = make_pipeline(covest,reg,ts)
    preprocess.fit(np.array(X_train_masked))

    X_ptrain = preprocess.transform(X_train_masked)
    
    X_test_masked = apply_cls_mask(X_test, M, k_s)
    X_ptest = preprocess.transform(X_test_masked)

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


def get_ablation_3_results_for_sub(num):
    fmris_transposed, stimuli, vt_mask_tensor = prepare_sub_data(num)

    X, s, y = ExtractSegments(fmris_transposed, stimuli, tau=19)
    y=y-1
    X = X.astype('float64')
    s = s.astype('int')
    X_train, X_test, s_train, _, y_train, y_test = train_test_split(X, s, y, test_size=0.20, random_state=42, stratify=y)

    s_train_new = s_train
    s_train_new[np.where(s_train_new!=0)] = 1
    s_train_new = np.concatenate(s_train_new)
    
    X_train_masked = apply_cls_mask(X_train, vt_mask_tensor, None)
    X_test_masked = apply_cls_mask(X_test, vt_mask_tensor, None)

    covest = Covariances()
    reg = Shrinkage(shrinkage=1e-3)
    ts = TangentSpace()
    preprocess = make_pipeline(covest,reg,ts)
    preprocess.fit(np.array(X_train_masked))

    X_ptrain = preprocess.transform(X_train_masked)
    
    X_test_masked = apply_cls_mask(X_test, vt_mask_tensor, None)
    X_ptest = preprocess.transform(X_test_masked)

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
