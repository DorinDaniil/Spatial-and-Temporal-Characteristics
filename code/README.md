## Code Directory Structure

- **`main.ipynb`**: This Jupyter notebook contains the primary experiments and analysis conducted for the project. It serves as the main entry point for running and evaluating the models.

- **`utils.py`**: This module includes utility functions and methods for visualization. It helps in creating plots and visual representations of the data and results.

- **`ablation.py`**: This script contains models and methods for ablation studies. It is used to analyze the impact of excluding certain components or features from the models.

- **`dataloader.py`**: This module implements the dataset and its associated methods. It also includes functionality for segmenting time series data, which is crucial for preprocessing and preparing the data for analysis.

- **`models.py`**: This file contains the main models used for classifying time series data. It includes the implementations of various classification algorithms tailored for time series analysis.

- **`requirements.txt`**: This file lists all the dependencies and libraries required to run the project. Ensure you have these installed in your environment before running the code.

## Getting Started

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/DorinDaniil/Spatial-and-Temporal-Characteristics.git
   cd code
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Main Notebook**:
   Open `main.ipynb` in a Jupyter Notebook environment and execute the cells to reproduce the experiments and analysis.
