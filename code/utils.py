import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import colors


# Сегментация временного ряда по классам стимулов
def extract_segments(X, s, tau):
    # Initialize lists to store the segments
    X_segments = []
    s_segments = []
    y = []

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
    X_segments = np.array(X_segments)
    s_segments = np.array(s_segments)
    y = np.array(y)

    return X_segments, s_segments, y


# Отрисовка слайсов
def save_scan_slices(scan, dim_slices, mask=None, title='Masked fMRI scan slices', save_img=False, img_path='.'):
    for dim, slice in enumerate(dim_slices):
        fig = plt.figure(figsize=(5, 5))
        if mask is None:
            if dim == 0:
                scan_slice = scan[slice, :, :].T
            elif dim == 1:
                scan_slice = scan[:, slice, :].T
            elif dim == 2:
                scan_slice = scan[:, :, slice].T
        else:
            if dim == 0:
                scan_slice = scan[slice, :, :].T
                scan_slice_masked = mask[slice, :, :].T
            elif dim == 1:
                scan_slice = scan[:, slice, :].T
                scan_slice_masked = mask[:, slice, :].T
            elif dim == 2:
                scan_slice = scan[:, :, slice].T
                scan_slice_masked = mask[:, :, slice].T

        plt.imshow(scan_slice, cmap="gray", origin="lower")
        plt.tick_params(labelsize=14)

        if mask is not None:
            cmap = colors.ListedColormap(['black', 'red'])
            plt.imshow(scan_slice_masked, cmap=cmap, origin="lower", alpha=0.5)

        if save_img:
            file_name = f"{img_path}/slice_dim_{dim}_slice_{slice}.png"
            os.makedirs(os.path.dirname(file_name), exist_ok=True)
            plt.savefig(file_name)
        plt.close()



def show_scan_slices(scan, dim_slices, mask=None, title='Masked fMRI scan slices', save_img=False, img_path='.'):
    fig = plt.figure(figsize=(15, 5))
    gs = fig.add_gridspec(1, 3, wspace=0.01)
    axes = gs.subplots(sharey=True)

    for dim, slice in enumerate(dim_slices):
        if mask is None:
            if dim == 0:
                scan_slice = scan[slice, :, :].T
            elif dim == 1:
                scan_slice = scan[:, slice, :].T
            elif dim == 2:
                scan_slice = scan[:, :, slice].T
        else:
            if dim == 0:
                scan_slice = scan[slice, :, :].T
                scan_slice_masked = mask[slice, :, :].T
            elif dim == 1:
                scan_slice = scan[:, slice, :].T
                scan_slice_masked = mask[:, slice, :].T
            elif dim == 2:
                scan_slice = scan[:, :, slice].T
                scan_slice_masked = mask[:, :, slice].T

        axes[dim].imshow(scan_slice, cmap="gray", origin="lower")
        axes[dim].set_title(f"Dim: {dim}, Slice: {slice}", fontsize=16)
        axes[dim].tick_params(labelsize=14)

        if mask is not None:
            cmap = colors.ListedColormap(['black', 'red'])
            axes[dim].imshow(scan_slice_masked, cmap=cmap, origin="lower", alpha=0.5)
            
    fig.suptitle(title, fontsize=18, y=1.02)
    plt.tight_layout()

    if save_img:
        file_name = f"{img_path}/slices.png"
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        plt.savefig(file_name)

    plt.show()