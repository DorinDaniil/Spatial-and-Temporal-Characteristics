import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import pandas as pd
from nilearn.datasets import fetch_haxby

# dataloader
class haxby_dataset():
    def __init__(self):
        path = '\data_fmri'
        self.data_files = fetch_haxby(data_dir=path, subjects=(1, 2, 3, 4, 5, 6), fetch_stimuli=True, url=None, resume=True, verbose=1)

    # Получение данных конкретного испытуемого
    def get_sub_data(self, num):
        # пути к данным
        fmri_data = self.data_files['func'][num-1]  # фМРТ-снимки
        labels = self.data_files['session_target'][num-1] # метки классов

        fmri_img = nib.load(fmri_data)
        fmri_array = fmri_img.get_fdata()
        labels_frame = pd.read_csv(labels, delimiter=' ')
        return fmri_array, labels_frame
    

    def plot_stimuli(self):
        stimulus_information = self.data_files.stimuli

        for stim_type in stimulus_information:
            # skip control images, there are too many
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