# 🌲 Credit Card Fraud Detection Using Random Forest

This project is a machine learning implementation of a **Random Forest Classifier** to detect fraudulent credit card transactions. It includes preprocessing, cross-validation, model training, evaluation, and visualization.

---

## 📁 Dataset

- **Filename:** `creditcard_2023.csv`
- **Target column:** `Class` (1 = Fraud, 0 = Legitimate)
- The `id` column is removed before training.

---

## 🔍 Project Highlights

- Built using Python, scikit-learn, pandas, seaborn, and matplotlib.
- Performs cross-validation using F1 score.
- Visualizes:
  - Confusion matrix
  - Feature importance
  - Correlation matrix
  - ROC curve with AUC score

---

## 🛠️ Tech Stack

- Python 3.x
- pandas
- numpy
- scikit-learn
- seaborn
- matplotlib

---

## 🚀 How to Run

### 1. Clone the repo:
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### 2. Install Required Packages:
```bash
pip install pandas numpy scikit-learn seaborn matplotlib
```

### 3. Run the Script

Make sure the `creditcard_2023.csv` file is in the same directory as the script.

```bash
python random_forest_fraud.py
```

---

## 📊 Model Details

**Classifier:** `RandomForestClassifier`

### 🔧 Parameters:

- `n_estimators = 100`
- `max_depth = 10`
- `min_samples_split = 5`
- `random_state = 42`

### 📏 Evaluation Metrics:

- Confusion Matrix  
- Classification Report  
- ROC AUC Score  
- Feature Importance

---

## 📈 Results

- **Cross-validated F1 Scores:**  
  *(Fill in your printed F1 scores here)*

- **AUC Score:**  
  *(Add your AUC score here)*

- **Top Features Influencing Prediction:**  
  *(List top 3–5 features based on importance plot)*

---

## 📌 Visuals Included

- ✅ Confusion Matrix  
- ✅ Feature Importance Bar Chart  
- ✅ Correlation Matrix Heatmap  
- ✅ ROC Curve

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change or improve.

---

## 📄 License

This project is licensed under the MIT License.
