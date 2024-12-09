<div align="center">
<h1>Enhancing fMRI Data Decoding with Spatio-Temporal
Characteristics in Limited Dataset Approaches </h1>

[Daniil Dorin](https://github.com/DorinDaniil)<sup>1 :email:</sup>, [Andrey Grabovoy](https://github.com/andriygav)<sup>1</sup>, [Vadim Strijov](https://github.com/Strijov)<sup>2</sup>

<sup>1</sup> Antiplagiat Company, Moscow, Russia

<sup>2</sup> Forecsys, Moscow, Russia

<sup>:email:</sup> Corresponding author

[📝 Paper](), [</> Code](https://github.com/DorinDaniil/Spatial-and-Temporal-Characteristics/tree/main/code)

![overview](https://github.com/user-attachments/assets/09233ad1-518e-484f-9ed0-11a2d79a1744)

</div>

## 💡 Abstract
This study investigates the impact of spatio-temporal features on the quality of decoding functional Magnetic Resonance Imaging (fMRI) data. Neural network architectures are limited in handling fMRI data due to the small data volume, high individual variability, and significant computational resources required. This makes it a crucial task to find a method for decoding with an insufficient dataset. Therefore, an approach considering the unique structural features of each subject's brain is examined. To build a solution we propose method for extracting a unique activity mask of the brain for each subject is proposed. It is based on reducing the spatial dimensionality of fMRI time series through weighting stimulated brain regions using the cross-correlation function. A classification model for fMRI time series data is developed, incorporating activity masks extracted for each activity class using the mentioned method and an encoder that employs Riemannian geometry to extract spatio-temporal characteristics. The computational experiment analyzes the proposed methods on a sample obtained from tomographic examinations of 6 subjects. An ablation analysis of the proposed classification method shows a significant decrease in quality when any component of the method is missing, highlighting the importance of the extracted features for high-quality fMRI data decoding.
## 🔎 Overview
<div align="center">
  
![scheme_english](https://github.com/user-attachments/assets/8a300875-9af6-41c4-b4ed-f8b0e5f6832f)

</div>

## 🛠️ Repository Structure
The repository is structured as follows:
- `assets`: The directory contains the main figures
- `code`: The directory contains the code used in the project
