import matplotlib.pyplot as plt
import os
from matplotlib import colors


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