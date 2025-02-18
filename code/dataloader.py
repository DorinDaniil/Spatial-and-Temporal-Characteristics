import matplotlib.pyplot as plt
import nibabel as nib
import pandas as pd
import numpy as np
from nilearn.image import load_img
from nilearn.datasets import fetch_haxby

class HaxbyDataset:
    def __init__(self, path='\data_fmri') -> None:
        """
        Initializes the Haxby dataset class.
        """
        self.data_files = fetch_haxby(data_dir=path, subjects=(1, 2, 3, 4, 5, 6), fetch_stimuli=True, url=None, resume=True, verbose=1)

    def get_sub_data(self, num: int) -> tuple:
        """
        Gets the data for a specific subject.

        Args:
        num (int): The subject number.

        Returns:
        tuple: A tuple containing the fMRI data and labels.
        """
        # File paths to data
        fmri_data = self.data_files['func'][num-1]  # fMRI scans
        labels = self.data_files['session_target'][num-1] # class labels

        fmri_img = nib.load(fmri_data)
        fmri_array = fmri_img.get_fdata()
        labels_frame = pd.read_csv(labels, delimiter=' ')
        return fmri_array, labels_frame

    def plot_stimuli(self) -> None:
        """
        Plots the stimuli.
        """
        stimulus_information = self.data_files.stimuli

        for stim_type in stimulus_information:
            # Skip control images, there are too many
            if stim_type != 'controls':

                file_names = stimulus_information[stim_type]
                file_names = file_names[0:16]
                fig, axes = plt.subplots(4, 4)
                fig.suptitle(stim_type)

                for img_path, ax in zip(file_names, axes.ravel()):
                    ax.imshow(plt.imread(img_path), cmap=plt.cm.gray)

                for ax in axes.ravel():
                    ax.axis("off")
        plt.show()

def prepare_sub_data(num):
    dataset = HaxbyDataset()
    data_files = dataset.data_files
    fmris, labels = dataset.get_sub_data(num)
    class_dict = {'rest': 0, 'bottle': 1, 'cat': 2, 'chair': 3, 'face': 4, 'house': 5, 'scissors': 6, 'scrambledpix': 7, 'shoe': 8}

    stimuli = np.array([class_dict[label] for label in labels['labels'].values])
    fmris_transposed = fmris.transpose((3, 0, 1, 2))
    vt_mask_filename = load_img(data_files.mask_vt[num-1])
    vt_mask_tensor = load_mask_from_nifti(vt_mask_filename)
    return fmris_transposed, stimuli, vt_mask_tensor


def load_mask_from_nifti(nifti_image):
    mask_data = nifti_image.get_fdata()
    mask_tensor = np.where(mask_data > 0, 1, 0)
    return mask_tensor