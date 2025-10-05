# 🧪 QSAR-Antimalarial-Model
![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![RDKit](https://img.shields.io/badge/RDKit-Enabled-lightgrey.svg)
![Build](https://img.shields.io/badge/Status-Active-brightgreen.svg)

---

## 🎯 Project Overview
Malaria remains a serious global health problem, particularly in sub-Saharan Africa, with *Plasmodium falciparum* responsible for most severe cases.  
This project applies **Quantitative Structure–Activity Relationship (QSAR)** modeling to predict the biological activity of antimalarial compounds using **machine learning**.  

By leveraging molecular descriptors computed from **RDKit** and **Mordred**, and training models such as **Random Forest**, **XGBoost**, and **Neural Networks**, the goal is to develop a predictive tool that can identify potential antimalarial leads prior to docking or synthesis.

---

## 🧩 Objectives
- Develop QSAR models that predict *Plasmodium falciparum* inhibitory activity.  
- Compute 2D descriptors using RDKit and Mordred.  
- Train and compare machine learning algorithms: Random Forest, XGBoost, and Neural Networks.  
- Evaluate models using ROC-AUC, accuracy, precision, and F1-score.  
- Provide interpretable insights on molecular features influencing antimalarial activity.  

---

## 🧠 Methodology

### 1. Dataset Collection
- Data retrieved from **ChEMBL** and/or **PubChem BioAssay** on antimalarial targets such as *Plasmodium falciparum DHFR* (e.g., CHEMBL240).  
- Bioactivity data (IC₅₀ / pIC₅₀ values) downloaded in CSV or SDF format.  
- Data cleaning and SMILES standardization performed using **RDKit**.

### 2. Descriptor Generation
- Calculated >1000 2D molecular descriptors using **Mordred** and **RDKit**.  
- Performed descriptor curation and feature selection (Variance Threshold, Correlation Filtering, Recursive Feature Elimination).  

### 3. Model Development
- Built and optimized ML models using **Scikit-learn** and **XGBoost**.  
- Algorithms tested:  
  - Random Forest Classifier  
  - XGBoost Classifier  
  - Feedforward Neural Network (Keras/TensorFlow)  
- Hyperparameter tuning with **RandomizedSearchCV** and **GridSearchCV**.

### 4. Model Evaluation
- Metrics: Accuracy, ROC-AUC, F1-score, Precision, Recall.  
- Visualization: Confusion Matrix, ROC Curve, Feature Importance plots.  
- Cross-validation (5-fold) for generalization performance.  

### 5. Prediction and Virtual Screening
- Used trained models to predict activity of new natural compounds (from PubChem or in-house phytochemical library).  
- Identified top candidates for subsequent docking and ADMET analysis.

---

## 🧰 Tools and Libraries

| Category | Tools/Packages |
|-----------|----------------|
| **Data Processing** | Pandas, NumPy |
| **Descriptor Generation** | RDKit, Mordred |
| **Modeling** | Scikit-learn, XGBoost, TensorFlow/Keras |
| **Visualization** | Matplotlib, Seaborn |
| **Environment** | Jupyter Notebook, Python 3.10+ |

---

## 📂 Project Structure
QSAR-Antimalarial-Model/
│── data/
│ ├── raw_data.csv
│ ├── processed_data.csv
│── notebooks/
│ ├── 01_data_preparation.ipynb
│ ├── 02_descriptor_generation.ipynb
│ ├── 03_model_training_evaluation.ipynb
│ ├── 04_prediction_visualization.ipynb
│── src/
│ ├── descriptors.py
│ ├── train_model.py
│── results/
│ ├── metrics_summary.csv
│ ├── feature_importance.png
│ ├── roc_curve.png
│── requirements.txt
│── LICENSE
│── README.md


---

## 📊 Example Results (Illustrative)

| Model | Accuracy | ROC-AUC | F1-score |
|--------|-----------|----------|-----------|
| Random Forest | 0.89 | 0.92 | 0.88 |
| XGBoost | 0.91 | 0.94 | 0.90 |
| Neural Network | 0.87 | 0.90 | 0.86 |

---

## 💡 Insights
- **logP**, **TPSA**, and **H-bond donors/acceptors** are among the most influential descriptors.  
- Ensemble models (Random Forest, XGBoost) performed best.  
- QSAR predictions can prioritize compounds before expensive docking and synthesis steps.  

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

✨ “Combining chemistry, computation, and curiosity to discover tomorrow’s medicines.”



