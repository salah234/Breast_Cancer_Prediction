
# Breast Cancer Prediction Model  
![Repository Badge Placeholder](https://img.shields.io/badge/Machine%20Learning-Project-blue)

## Table of Contents
1. [Introduction](#introduction)  
2. [Dataset](#dataset)  
3. [Technologies Used](#technologies-used)  
4. [Models](#model-development)  
5. [Performance Metrics](#performance-metrics)  
6. [Repository Structure](#repository-structure)  
7. [Setup and Installation](#setup-and-installation)  
9. [Contributing](#contributing)  


---

## Introduction  
Breast cancer remains one of the most prevalent and life-threatening diseases affecting women worldwide. Early diagnosis and accurate predictions can significantly improve survival rates. This repository showcases a machine learning solution for predicting breast cancer using a neural network and comparative analysis with other classification algorithms.

Key Highlights:  
- Developed using the **University of Wisconsin Breast Cancer Dataset**.  
- Neural Network model achieved an accuracy of **99.12%**.  
- Comparative analysis with **Random Forest (99%)** and **XGBoost (97%)**.

---

## Dataset  
We utilized the **University of Wisconsin Breast Cancer Dataset**, which contains detailed features extracted from digitized images of fine needle aspirate (FNA) of breast mass.  

### Features:  
- **'mean radius' 'mean texture' 'mean perimeter' 'mean area'
 'mean smoothness' 'mean compactness' 'mean concavity'
 'mean concave points' 'mean symmetry' 'mean fractal dimension'
 'radius error' 'texture error' 'perimeter error' 'area error'
 'smoothness error' 'compactness error' 'concavity error'
 'concave points error' 'symmetry error' 'fractal dimension error'
 'worst radius' 'worst texture' 'worst perimeter' 'worst area'
 'worst smoothness' 'worst compactness' 'worst concavity'
 'worst concave points' 'worst symmetry' 'worst fractal dimension'**, and more.  
- Labels: **Benign** (0) or **Malignant** (1).  

### Dataset Reference:  
[UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)  

---

## Technologies Used  
- **Python 3.10+**  
- **TensorFlow/Keras** (Neural Network)  
- **Scikit-learn** (Random Forest, XGBoost)  
- **Matplotlib** and **Seaborn** (Visualizations)  
- **Pandas and NumPy** (Data Processing)

---

### Neural Network Architecture  
- **Input Layer:** 30 features (normalized).  
- **Hidden Layers:** Two dense layers with ReLU activation.  
- **Output Layer:** Sigmoid activation for binary classification.  

### Comparative Models:  
1. **Random Forest Classifier:**  
   - Grid-searched hyperparameters for optimal performance.  
   - Achieved **99% accuracy**.  
2. **XGBoost Classifier:**  
   - Tuned for depth, learning rate, and boosting rounds.  
   - Achieved **97% accuracy**.  

---

## Performance Metrics  
| Model                 | Accuracy (%) | Precision | Recall | F1 Score |  
|-----------------------|-------------|-----------|--------|----------|  
| Neural Network        | 99.12       | 0.96      | 0.97   | 0.97     |  
| Random Forest         | 99.00       | 0.99      | 1.00   | 0.99     |  
| XGBoost               | 97.00       | 0.97      | 0.99   | 0.98     |  

**Neural Network**  
> <img width="442" alt="Screenshot 2024-11-21 at 2 56 08 PM" src="https://github.com/user-attachments/assets/1bc52f69-50da-43a9-a6ae-c0f089b2be55">
**Random Classification Model**  
> <img width="443" alt="Screenshot 2024-11-21 at 2 57 06 PM" src="https://github.com/user-attachments/assets/bba66fe9-283f-498d-90be-a6095f27e4a4">
**XGboast Algorithm**  
> <img width="443" alt="Screenshot 2024-11-21 at 2 57 06 PM" src="https://github.com/user-attachments/assets/bba66fe9-283f-498d-90be-a6095f27e4a4">

---
**Visualization for the Sharpley Values**
> <img width="624" alt="Screenshot 2024-11-21 at 4 32 54 PM" src="https://github.com/user-attachments/assets/77729928-c84b-429a-aa21-a9c5f295467b">

## Repository Structure  
```plaintext  
Breast-Cancer-Prediction/  
├── data/  
│   └── breast_cancer_data.csv   # Dataset used  
├── models/  
│   ├── neural_network_model.py  # Neural network implementation  
│   ├── random_forest_model.py   # Random forest implementation  
│   └── xgboost_model.py         # XGBoost implementation  
├── notebooks/  
│   └── EDA.ipynb                # Exploratory Data Analysis notebook  
├── visuals/  
│   ├── confusion_matrix.png     # Confusion Matrix visual  
│   └── roc_curve.png            # ROC Curve visual  
├── README.md                    # Project documentation  
└── requirements.txt             # Python dependencies  
```

---

## Setup and Installation  
Follow these steps to set up and run the project locally:

1. Clone the repository:  
   ```bash  
   git clone https://github.com/username/Breast-Cancer-Prediction.git  
   cd Breast-Cancer-Prediction  
   ```   

2. Run the Jupyter Notebook to execute the model scripts:  
   ```bash  
   python models/neural_network_model.py  
   ```

---
## Contributors
Shoutouts to
1. Salah Mohammed
2. Abbot Tubeine
3. Azeezat Akinboro
4. Alexia Rendon

## Contributing  
We welcome contributions! Please follow these steps:  
1. Fork the repository.  
2. Create a feature branch:  
   ```bash  
   git checkout -b feature-name  
   ```  
3. Commit changes and open a pull request.  

---

## Acknowledgments  
Special thanks to the University of Wisconsin for providing the dataset and the open-source community for tools like TensorFlow and Scikit-learn.  
