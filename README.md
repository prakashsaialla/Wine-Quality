
# 🍷 Wine Quality Prediction

This project uses **Machine Learning** to predict the quality of wine based on its physicochemical properties (like acidity, sugar, pH, alcohol content, etc.). The dataset is a well-known benchmark dataset from the UCI Machine Learning Repository.

## 🚀 Features

* Data preprocessing and cleaning
* Exploratory Data Analysis (EDA) with visualizations
* Feature engineering for better insights
* Machine Learning models (Logistic Regression, Random Forest, XGBoost, etc.)
* Model evaluation with metrics like **Accuracy, Precision, Recall, F1-Score**
* Predict wine quality (score from 0–10)

## 🛠️ Tech Stack

* **Python 3.x**
* **Pandas, NumPy** – data handling
* **Matplotlib, Seaborn** – visualization
* **Scikit-learn** – ML models & evaluation metrics
* **XGBoost / LightGBM** – boosting algorithms for accuracy

## 📂 Workflow

1. **Load Dataset** – Wine Quality dataset (red & white variants)
2. **Data Preprocessing**

   * Handle missing values (if any)
   * Normalize/scale numerical features
   * Encode categorical values (if needed)
3. **Exploratory Data Analysis (EDA)**

   * Correlation heatmaps
   * Distribution plots for features (alcohol, acidity, etc.)
   * Quality distribution of wines
4. **Model Training**

   * Split dataset into train/test sets
   * Train models (Logistic Regression, Random Forest, XGBoost, etc.)
   * Tune hyperparameters
5. **Evaluation**

   * Compare models with metrics (accuracy, precision, recall, F1, confusion matrix)
6. **Prediction**

   * Input new wine data → predict quality score

## ▶️ How to Run

1. Clone this repository

   ```bash
   git clone https://github.com/prakashsaialla/wine-quality-prediction.git
   cd wine-quality-prediction
   ```

2. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter Notebook

   ```bash
   jupyter notebook Wine-Quality.ipynb
   ```

4. Example prediction:

   ```python
   input_data = [[7.4, 0.7, 0.0, 1.9, 0.076, 11, 34, 0.9978, 3.51, 0.56, 9.4]]
   model.predict(input_data)
   # Output: [5]  -> Wine quality score
   ```

## 📊 Example Results

* Logistic Regression: Accuracy = 65%
* Random Forest: Accuracy = 72%
* XGBoost: Accuracy = 75% (best model)

## 📌 Future Improvements

* Try deep learning models for regression/classification
* Deploy as a **Flask/Django web app** for interactive predictions
* Improve accuracy with feature selection & ensemble models
* Add visualization dashboard for wine producers

## 📜 License

This project is licensed under the MIT License – feel free to use and modify.

