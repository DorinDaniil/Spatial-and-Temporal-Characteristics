<div align="center">
<h1>Enhancing fMRI Data Decoding with Spatiotemporal
Characteristics in Limited Dataset </h1>

[Daniil Dorin](https://github.com/DorinDaniil)<sup>1 :email:</sup>, [Andrey Grabovoy](https://github.com/andriygav)<sup>1</sup>, [Vadim Strijov](https://github.com/Strijov)<sup>2</sup>

<sup>1</sup> Antiplagiat Company, Moscow, Russia

<sup>2</sup> Forecsys, Moscow, Russia

<sup>:email:</sup> Corresponding author

[📝 Paper](https://www.mathnet.ru/php/archive.phtml?wshow=paper&jrnid=danma&paperid=664&option_lang=eng), [</> Code](https://github.com/DorinDaniil/Spatial-and-Temporal-Characteristics/tree/main/code), [🎬 Video](https://drive.google.com/file/d/13XwQ1Vhpog0QV_kDNLxPBhvNRJp__8BN/view?usp=sharing), [🎫 Poster](https://github.com/DorinDaniil/Spatial-and-Temporal-Characteristics/blob/main/assets/aij_poster.pdf)


![Figure_1_page-0001](https://github.com/user-attachments/assets/b008f21a-467d-41c4-b652-b894c0d5617c)

</div>

## 💡 Abstract
This study investigates the impact of spatiotemporal characteristics on the quality of the decoding of functional Magnetic Resonance Imaging (fMRI) data. Neural network architectures are limited in handling fMRI data due to the small sample size, high sample variability, and significant computational resources required. We propose a methodology for fMRI decoding with an insufficient dataset. We examine the unique structural features of each subject’s brain. To build a decoding methodology, we propose an algorithm for extracting a unique activity mask from the brain for each subject. This algorithm reduces the spatial dimensionality of fMRI time series through weighting stimulated brain regions using cross-correlation. We developed a classification model for the fMRI time series data from a single subject. It combines two parts: the first part is the brain activity masks extracted for each activity class using the extraction algorithm. The second part is an encoder that employs Riemannian geometry to extract spatiotemporal characteristics. The computational experiment analyzes the proposed methodology on a sample obtained from tomographic examinations of six subjects. An ablation analysis of the proposed classification model shows a significant decrease in quality when any component of the model is missing. Comparisons with neural network-based approaches demonstrate  the superior performance of the proposed methodology, especially in scenarios with limited data.
## 🔎 Overview
<div align="center">

![Figure_2_page-0001](https://github.com/user-attachments/assets/43d419cb-fe7a-410d-8091-dd8d35398b6d)

</div>

## 🛠️ Repository Structure
The repository is structured as follows:
- `assets`: The directory contains the main figures
- `code`: The directory contains the code used in the project

## :fire: Contributing

Feel free to contribute to this project by opening issues or submitting pull requests. Any feedback or suggestions for improvement are welcome.

