# 🧪 # Computational QSAR Analysis of Bioactive Antimalarial Constituents from *African Newbouldia laevis*
---
## 🎯 Project Overview
Malaria remains a serious global health problem, particularly in sub-Saharan Africa, with *Plasmodium falciparum* responsible for the most severe cases.  
This project applies **Quantitative Structure–Activity Relationship (QSAR)** modeling to predict the biological activity of antimalarial compounds using **machine learning**.  

By leveraging molecular descriptors computed from **RDKit** and **Extended-connectivity fingerprints (ECFPs)**, and training models such as **Random Forest**, **XGBoost**, and **CATBoost**,**LIGHTGBM**, and **LOGISTIC_REGRESSION** the goal is to develop a predictive tool that can identify potential antimalarial leads prior to Molecular docking or synthesis.

---

## 🧩 Objectives
- Develop QSAR models that predict *Plasmodium falciparum PDB ID: 7F3Y* inhibitory activity.  
- Compute 2D descriptors using RDKit and ECFP.  
- Train and compare machine learning algorithms: Random Forest, XGBoost, and Neural Networks.  
- Evaluate models using ROC-AUC, AUPRC, balanced accuracy, accuracy, recall,  precision, specificity, Matthews correlation coefficient, and F1-score.  
- Provide interpretable insights on molecular features influencing antimalarial activity.  

---

## 🧠 Methodology

### 1. Dataset Collection
- Data retrieved from **ChEMBL** and/or **PubChem BioAssay** on antimalarial targets such as *Plasmodium falciparum DHFR* (e.g., CHEMBL240).  
- Bioactivity data (pIC₅₀ values) downloaded in CSV or SDF format.  
- Data cleaning and SMILES standardization performed using **RDKit**.

### 2. Descriptor Generation
- Calculated >1000 2D molecular descriptors using **ECFP** and **RDKit**.  
- Performed descriptor curation and feature selection (Variance Threshold, Correlation Filtering, Recursive Feature Elimination).  

### 3. Model Development
- Built and optimized ML models using **Scikit-learn** and **XGBoost**.  
- Algorithms tested:  
  - Random Forest Classifier  
  - XGBoost Classifier
  - CATBoost Classifier
  - LightGBM Classifier
  - Logistic regression
- Hyperparameter tuning with  **GridSearchCV**.

### 4. Model Evaluation
- Metrics: Accuracy, ROC-AUC, F1-score, Precision, Recall.  
- Visualization: Confusion Matrix, ROC Curve, Feature Importance plots.  
- Cross-validation (5-fold) for generalization performance.  

### 5. Prediction and Virtual Screening
- Used trained models to predict the activity of new natural compounds (from Newbouldis Leavis).  


---

## 🧰 Tools and Libraries

| Category | Tools/Packages |
|-----------|----------------|
| **Data Processing** | Pandas, NumPy |
| **Descriptor Generation** | RDKit, Mordred |
| **Modeling** | Scikit-learn, XGBoost, RandomForest, Logistic Regression, CATBoost, LightGBM |
| **Visualization** | Matplotlib, Seaborn |
| **Environment** | Jupyter Notebook, Python 3.10+ |

---

## 📂 Project Structure
QSAR-Antimalarial-Model/
│── data/
│ ├── raw_data_chembl.csv
│ ├── processed_Chembl_data.csv
│── notebooks/
│ ├── Logistic_regression_qsar_model.ipynb
  ├── RandomForest_qsar_model.ipynb
│ ├── CatBoost_qsar_model.ipynb
│ ├── lightGBM_qsar_model.ipynb
│ ├── XGBoost_qsar_model.ipynb
│── 
│ ├── 
│ ├── 
│── results/
│ ├── confusion_matrix.csv
│ ├── feature_importance.png
│ ├── roc_curve and auprc.png
│── requirements.txt
│── LICENSE
│── README.md


---

## Models and Performance

The performance of five machine learning models on the test dataset is summarized below:

| Model               | Test Accuracy | Test F1-score | ROC AUC | AUPRC |
| ------------------- | ------------- | ------------- | ------- | ----- |
| CatBoost            | 0.6324        | 0.6135        | 0.74    | 0.61  |
| XGBoost             | 0.6324        | 0.6078        | 0.71    | 0.55  |
| RandomForest        | 0.6176        | 0.5952        | 0.75    | 0.64  |
| Logistic Regression | 0.6912        | 0.6225        | 0.68    | 0.56  |
| LightGBM            | 0.6324        | 0.6078        | 0.76    | 0.63  |

**Interpretation:**  
- While Logistic Regression has the highest test accuracy and F1-score, **LightGBM achieves the highest ROC AUC and competitive AUPRC**, making it more suitable for **ranking and separating classes**, which aligns with the primary goal of this study.  
- LightGBM is therefore selected as the **best model** for predicting antimalarial activity of compounds from *Newbouldia laevis*.

---

## Methodology

1. **Data Preparation:** Extracted molecular descriptors from the chemical constituents of *Newbouldia laevis*.  
2. **Model Training:** Trained multiple ML models including CatBoost, XGBoost, RandomForest, Logistic Regression, and LightGBM.  
3. **Evaluation:** Used test accuracy, F1-score, ROC AUC, and AUPRC to evaluate model performance.  
4. **Prediction:** Applied the selected LightGBM model to predict activity scores for all compounds, enabling ranking and prioritization.

---

## 🔬 Future Work
- Extend to multi-target prediction (DHFR, PfATPase, PfLDH).  
- Integrate docking and MM-GBSA scoring for deeper validation.  
- Develop a Streamlit app for interactive QSAR predictions.  

---

## 📘 References
- ChEMBL Database – [https://www.ebi.ac.uk/chembl/](https://www.ebi.ac.uk/chembl/)  
- RDKit Documentation – [https://www.rdkit.org/](https://www.rdkit.org/)  
- Mordred Descriptors – [https://github.com/mordred-descriptor/mordred](https://github.com/mordred-descriptor/mordred)
- ![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
- ![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
- ![RDKit](https://img.shields.io/badge/RDKit-Enabled-lightgrey.svg)
- ![Build](https://img.shields.io/badge/Status-Active-brightgreen.svg)

---

## 🧑‍🔬 Citation
If you use or reference this repository:
Izuagba, F.O. (2025). QSAR Modeling for Antimalarial Compounds. GitHub Repository.


---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/QSAR-Antimalarial-Model.git
cd QSAR-Antimalarial-Model

# Install dependencies
pip install -r requirements.txt
```

🧭 Contact

📧 izuagbafavour@gmail.com

🔗 LinkedIn: linkedin.com/in/izuagbafavourogadinma

🏢 Affiliation: Chemsolvers Research and Computational Laboratories
    Federal University of Technology, Owerri, Imo State, Nigeria

✨ “Combining chemistry, computation, and curiosity to discover tomorrow’s medicines.”



