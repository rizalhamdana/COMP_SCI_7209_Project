# COMP_SCI_7209_Project

# Project Instructions

This repository contains multiple Jupyter Notebooks for big data analysis and project assignments.

## Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd <your-repo-folder>

2. Create virtual environment

```
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate    

```

3. Install Dependencies

```
pip install -r requirements.txt

```

4. Launch Jupyter Notebook

```
jupyter notebook

```


## Usage

- Open any notebook (.ipynb file) in the repository.

- Run the cells sequentially to reproduce results or experiments.

- Modify and experiment with the code as needed.

This assignmets contains multiple part, where each part has several jupyter notebooks file. Herewith the details:

### Assignment Structure

| Part | Jupyter Notebooks | Description |
|------|--------------------|-------------|
| 1A   | *(No Jupyter Notebooks)* | No notebooks provided for this part. | Contains data correlation checks, completeness verification, and state selection process. |
| 1B   | - `correlation_analysis.ipynb` <br> - `data_completeness_check.ipynb` <br> - `states_selection.ipynb` |  Contains data correlation checks, completeness verification, and state selection process. |
| 1C   | - `model_selection.ipynb` <br> - `model_selection_selected_features.ipynb` | Model selection experiments: <br>• `model_selection.ipynb` → complete feature set <br>• `model_selection_selected_features.ipynb` → crop-specific feature selection |
| 1D   | *(No Jupyter Notebooks)* | No notebooks provided for this part. |


## Prerequisites

Before running the notebooks, ensure you have the following installed:

- **Python 3.10.18**
- **Jupyter Notebook** or **JupyterLab**
- Required Python libraries (see `requirements.txt`):
  - matplotlib==3.10.5  
  - numpy==2.3.2  
  - pandas==2.3.1  
  - scikit-learn==1.7.1  
  - scipy==1.16.1  
  - seaborn==0.13.2  