# SVM-Based Cancer Classification

This project demonstrates the use of **Support Vector Machines (SVM)** for binary classification using the Breast Cancer dataset. It includes both **linear** and **non-linear (RBF kernel)** SVM models, decision boundary visualizations, and hyperparameter tuning with `GridSearchCV`.

---

## 🔍 Objective

To classify tumor diagnosis (Malignant or Benign) using SVM techniques with the following goals:

1. Prepare and preprocess the dataset.
2. Train SVM models using linear and RBF kernels.
3. Visualize decision boundaries.
4. Evaluate model performance.
5. Tune hyperparameters (`C` and `gamma`) using cross-validation.

---

## 📂 Dataset

- **File:** `Cancer_Data.csv`
- **Target Column:** `diagnosis`
  - `'M'` = Malignant (mapped to 1)
  - `'B'` = Benign (mapped to 0)
- **Features Used:** `radius_mean`, `texture_mean`

---

## 🛠️ Tools and Libraries

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

---

## 🧪 Model Training

### ✅ Linear SVM
```python
model = SVC(kernel='linear')
model.fit(x_train, y_train)

 **## RBF **
rbf = SVC(kernel='rbf', gamma='scale')
rbf.fit(x_train, y_train)
