# Customer Churn Prediction Project

**Project Overview:** This project focuses on predicting customer churn, which refers to the event when a customer leaves a service. In the telecom industry, churn prediction helps identify at-risk customers and allows companies to implement proactive retention strategies (e.g., targeted offers), which is more cost-effective than acquiring new customers. By developing a machine learning model trained on historical data, we can predict which customers are likely to cancel their service, allowing businesses to take actions to prevent churn, ultimately improving both revenue and customer satisfaction.

**Features:** The codebase implements a typical churn analysis pipeline:

* **Data Preprocessing:** Cleaning the raw data (handling missing or inconsistent values, encoding categorical fields, normalizing or scaling numeric features, etc.).
* **Exploratory Data Analysis (EDA):** Generating summary statistics and visualizations (e.g. histograms, boxplots, heatmaps) to understand feature distributions and identify patterns related to churn.
* **Model Training:** Using libraries like scikit-learn to train classification models (e.g. Logistic Regression, Random Forest, SVM) on the prepared data. Feature engineering (one-hot encoding, feature selection) and handling class imbalance (resampling) are also applied.
* **Model Evaluation:** Evaluating performance with metrics such as accuracy, precision, recall, F1‐score, and ROC‐AUC on hold-out or cross-validated test sets. For example, one study on the Telco churn dataset found the best classifiers achieved around **80% accuracy**.
* **Prediction:** Using the final model to predict churn probabilities for new or unseen customer data, enabling the business to act on the highest-risk cases.

**Dataset:** We use the **Customer Churn** dataset (an Hugging Face dataset). This is a data with ~24k customer records. Each row represents a customer, and the columns include:

* A **Churn** indicator (whether the customer left the service in the last month).
* **Service details:** types of services the customer subscribes to (e.g. phone, internet, streaming).
* **Account info:** tenure (how long they’ve been a customer), billing contract, payment method, monthly charges, total charges.
* **Demographics:** customer gender, age bracket, and whether they have partners or dependents.

This rich feature set (with 21+ columns) allows the model to learn patterns correlated with churn.

**Technologies Used:** The project is implemented in **Python** (typically with Jupyter Notebooks). Key libraries and tools include:

* **Data handling and analysis:** `pandas`, `numpy` (for data frames and numerical operations).
* **Visualization:** `matplotlib`, `seaborn` for charts and plots (histograms, boxplots, heatmaps).
* **Machine Learning:** `scikit-learn` for modeling and evaluation (classification algorithms, train/test split, metrics).  Optionally, `imbalanced-learn` (`imblearn`) can be used for oversampling or other resampling techniques if classes are imbalanced.
* **Other tools:** Version control with Git/GitHub, and any required utility libraries (e.g. `pickle` for model saving, `joblib`, etc.).

**How to Run:** To set up and execute the analysis:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/1varma/CustomerChurn.git
   cd CustomerChurn
   ```

2. **Install dependencies:** Ensure you have Python installed (3.7+). Then install required packages, e.g.:

   ```bash
   pip install -r requirements.txt
   ```

   *(This installs pandas, numpy, scikit-learn, matplotlib, seaborn, etc. as listed in `requirements.txt`.)*

3. **Launch the analysis:** Open the provided Jupyter notebook (e.g. `Customer_Churn.ipynb`) or run the main script. For notebooks, you can start Jupyter:

   ```bash
   jupyter notebook
   ```

   Then open the churn analysis notebook and run the cells in order. For scripts, use a command like `python churn_model.py` (depending on how the code is organized).

4. **View results:** The code will output evaluation metrics and save any plots to the `Results/` or similar folder. You can inspect these outputs (confusion matrix, ROC curves, feature importance charts, etc.) to understand model performance.

**Model Performance:**  After training and testing, we summarize the key results. For example, the best models often achieve around **80% classification accuracy** on the test set. Metrics like **precision/recall** and **ROC-AUC** are used to balance false positives/negatives.  In practice, a high recall on the “churn” class is desirable to catch most at-risk customers, even if that means investigating some false alarms.  The evaluation includes confusion matrices and classification reports for transparency. These results confirm that the model can reliably distinguish churners, enabling targeted retention efforts.

**Screenshots or Outputs:** The repository includes visual outputs illustrating the analysis.  Examples might include:

* Churn vs. tenure or contract type bar charts (showing higher churn rates for short-term contracts).
* Correlation heatmaps or pair plots of features.
* Model evaluation plots (such as ROC curves or feature importance bar charts).

(Refer to the `Results/` or `visuals/` folder in the repo for these figures.) These charts help communicate findings to both technical reviewers and business stakeholders.

**Acknowledgments:** This project is based on the **Telco Customer Churn** dataset originally published by IBM (available on Kaggle). We thank the data providers for this benchmark dataset. The analysis and code were inspired by standard churn prediction tutorials and kaggle examples, which helped shape the approach.

**References:** Key information and best practices were drawn from industry and academic sources on churn prediction, as well as the Kaggle documentation for the IBM Telco Customer Churn dataset.   (Please see the inline citations above for specific references.)
