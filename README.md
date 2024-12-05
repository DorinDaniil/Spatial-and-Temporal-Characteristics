<div align="center">
<h1>Enhancing fMRI Data Decoding with Spatio-Temporal
Characteristics in Limited Dataset Approaches </h1>

[Daniil Dorin](https://github.com/DorinDaniil)<sup>1 :email:</sup>, [Andrey Grabovoy](https://github.com/andriygav)<sup>1</sup>, [Vadim Strijov](https://github.com/Strijov)<sup>2</sup>

<sup>1</sup> Antiplagiat Company, Moscow, Russia

<sup>2</sup> Forecsys, Moscow, Russia

<sup>:email:</sup> Corresponding author

[📝 Paper](), [</> Code](https://github.com/DorinDaniil/Spatial-and-Temporal-Characteristics/tree/main/code)
</div>

## 💡 Abstract
This study investigates the impact of spatio-temporal features on the quality of
decoding functional Magnetic Resonance Imaging (fMRI) data with discrete time
representation. Neural network architectures are limited in handling fMRI data
due to the small data volume, high individual variability, and significant computa-
tional resources required. Therefore, an approach considering the unique structural
features of each subject’s brain is examined. The core objective is to evaluate the
classification quality in the context of a small dataset, where the available data
is insufficient for training neural networks effectively. A method for extracting a
unique activity mask of the brain for each subject is proposed. It is based on reduc-
ing the spatial dimensionality of fMRI time series through weighting stimulated
brain regions using the cross-correlation function. A classification model for fMRI
time series is developed, utilizing activity masks extracted using the mentioned
method for each activity class and an encoder employing Riemannian geometry
to extract spatio-temporal characteristics. The computational experiment analyzes
the proposed methods on a sample obtained from tomographic examinations of
6 subjects. An ablation analysis of the proposed classification method shows a
significant decrease in quality when any component of the method is missing,
highlighting the importance of the extracted features for high-quality fMRI data
decoding.
## 🔎 Overview
<div align="center">

</div>

## 🛠️ Repository Structure
The repository is structured as follows:
- `assets`: The directory contains the main figures
- `code`: The directory contains the code used in the project
