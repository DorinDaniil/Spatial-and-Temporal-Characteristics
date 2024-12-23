<div align="center">
<h1>Enhancing fMRI Data Decoding with Spatio-Temporal
Characteristics in Limited Dataset </h1>

[Daniil Dorin](https://github.com/DorinDaniil)<sup>1 :email:</sup>, [Andrey Grabovoy](https://github.com/andriygav)<sup>1</sup>, [Vadim Strijov](https://github.com/Strijov)<sup>2</sup>

<sup>1</sup> Antiplagiat Company, Moscow, Russia

<sup>2</sup> Forecsys, Moscow, Russia

<sup>:email:</sup> Corresponding author

[📝 Paper](), [</> Code](https://github.com/DorinDaniil/Spatial-and-Temporal-Characteristics/tree/main/code)

![Figure_1-1](https://github.com/user-attachments/assets/70f39ca6-3051-419f-9186-d08b34fab894)

</div>

## 💡 Abstract
This study investigates the impact of spatiotemporal features on the quality of the decoding of functional Magnetic Resonance Imaging (fMRI) data. Neural network architectures are limited in handling fMRI data due to the small sample size, high sample variability, and significant computational resources required. We have to find a methodology for fMRI decoding with an insufficient dataset. We examine the unique structural features of each subject’s brain. To build a decoding methodology, we propose a method for extracting a unique activity mask from the brain for each subject. This method reduces the spatial dimensionality of fMRI time series through weighting stimulated brain regions using cross-correlation. We developed a classification model for the fMRI time series data from a single subject. It combines two parts. The first part is the brain activity masks extracted for each activity class using the extraction method. The second part is an encoder that employs Riemannian geometry to extract spatiotemporal characteristics. The computational experiment analyzes the proposed method on a sample obtained from tomographic examinations of six subjects. An ablation analysis of the proposed classification method shows a significant decrease in quality when any component of the method is missing. It highlights the importance of extracted features for high-quality fMRI data decoding.
## 🔎 Overview
<div align="center">

![Figure_2-1](https://github.com/user-attachments/assets/700f2814-b822-40f2-bfe0-3e381e147fe8)

</div>

## 🛠️ Repository Structure
The repository is structured as follows:
- `assets`: The directory contains the main figures
- `code`: The directory contains the code used in the project
