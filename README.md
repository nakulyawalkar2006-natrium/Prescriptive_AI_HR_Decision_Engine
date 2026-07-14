# 🏢 Enterprise HR Retention AI

An end-to-end **Machine Learning application** that predicts employee attrition and provides **AI-powered retention recommendations** through an interactive **Streamlit dashboard**. The project combines predictive analytics, feature engineering, explainable AI, and cost-benefit analysis to help HR teams identify at-risk employees and make data-driven retention decisions.

---

## 🚀 Features

- 🤖 Employee Attrition Prediction using Machine Learning
- 📊 Interactive Streamlit Dashboard
- 🧠 AI Explainability for individual employees
- 💰 Financial Retention Cost Analysis
- 🧪 "What-If" HR Intervention Simulator
- 📈 Company-wide Flight Risk Heatmap
- ⚙️ Automated Data Preprocessing Pipeline
- 🎯 Feature Selection using Random Forest
- 🔄 Class Balancing with SMOTE
- 📉 Dynamic Decision Threshold Optimization

---

# 📸 Dashboard Features

## 👤 Individual Employee Analysis

- Predict employee flight risk
- View expected financial loss
- AI explanation of important risk factors
- Simulate salary raises
- Modify work environment satisfaction
- Test overwork reduction scenarios

---

## 🗺️ Company-wide Analytics

- Flight risk heatmap
- Average risk by job role
- Organization-wide attrition insights
- Department comparison

---

# 🏗️ Project Architecture

```
                HR Dataset
                     │
                     ▼
          Data Preprocessing Pipeline
                     │
                     ▼
          Feature Engineering
                     │
                     ▼
           One-Hot Encoding
                     │
                     ▼
               Train/Test Split
                     │
                     ▼
                  SMOTE
                     │
                     ▼
          Random Forest Feature Selection
                     │
                     ▼
      Random Forest Classification Model
                     │
                     ▼
     Threshold Optimization (Maximum F1)
                     │
                     ▼
         Streamlit Interactive Dashboard
                     │
         ┌───────────┴────────────┐
         ▼                        ▼
 Employee Analysis         Company Analytics
```

---

# 📂 Project Structure

```
.
├── app.py                     # Streamlit application
├── model.py                   # Model training pipeline
├── data_pipeline.py           # Data preprocessing
├── retention_check.py         # Financial decision engine
├── model.ipynb                # Model experimentation notebook
├── processed_attrition_data.csv
├── corporate_config.json
├── requirements.txt
└── README.md
```

---

# 📊 Machine Learning Pipeline

### 1. Data Preprocessing

The preprocessing pipeline performs:

- Currency conversion (USD → INR)
- Annual CTC calculation
- Compensation Ratio computation
- Overwork Strain feature engineering
- Role Friction feature engineering
- Removal of unnecessary columns

---

### 2. Feature Engineering

Additional business-oriented features include:

- Annual CTC (LPA)
- Compensation Ratio
- Overwork Strain
- Role Friction

These engineered features improve model performance and make predictions more interpretable.

---

### 3. Data Preparation

- Label Encoding
- One-Hot Encoding
- Train/Test Split
- SMOTE Oversampling

---

### 4. Feature Selection

A baseline Random Forest identifies the most informative features.

Only features above the median importance threshold are retained before training the final model.

---

### 5. Machine Learning Model

The final model uses a tuned **Random Forest Classifier** with:

- Balanced class weights
- Maximum depth optimization
- Feature bagging
- Bootstrap disabled
- Optimized split parameters

---

### 6. Threshold Optimization

Instead of using the default 0.5 threshold, the model finds the probability threshold that maximizes the **F1 Score**, resulting in better performance on imbalanced datasets.

---

# 💰 Economic Decision Engine

The application doesn't just predict attrition—it also evaluates whether retaining an employee is financially worthwhile.

For every employee it calculates:

- Expected replacement cost
- Expected financial loss
- Cost of retention (salary increase)
- Potential savings
- Recommended HR action

Possible recommendations:

- ✅ Stable
- 💵 Give Raise
- 🚪 Allow Exit

---

# 🧪 What-If Simulator

HR managers can simulate interventions such as:

- Increase employee salary
- Reduce overwork strain
- Improve workplace satisfaction

The application instantly recalculates the employee's attrition probability, allowing HR teams to evaluate different retention strategies before making decisions.

---

# 🧠 AI Explainability

For each employee, the application compares key attributes against the average values of employees who stayed with the company.

It highlights:

- Most influential features
- Employee's value
- Average safe value
- Percentage deviation

This helps HR understand *why* the model predicts a high attrition risk.

---

# 📊 Company Analytics

The dashboard includes organization-wide insights such as:

- Average flight risk by job role
- Flight risk visualization
- Dangerous attrition threshold
- Comparative risk analysis across departments

---

# 🛠️ Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Matplotlib
- Seaborn

---

# ⚙️ Machine Learning Techniques

- Random Forest Classifier
- SMOTE Oversampling
- Feature Selection
- Label Encoding
- One-Hot Encoding
- Precision-Recall Curve
- Threshold Optimization
- Explainable AI

---

# ▶️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/hr-retention-ai.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the preprocessing pipeline:

```bash
python data_pipeline.py
```

Launch the Streamlit application:

```bash
streamlit run app.py
```

---

# 📈 Workflow

```
Raw HR Dataset
      │
      ▼
Data Cleaning
      │
      ▼
Feature Engineering
      │
      ▼
Model Training
      │
      ▼
Risk Prediction
      │
      ▼
Economic Analysis
      │
      ▼
Interactive Dashboard
```

---

# 🎯 Future Improvements

- SHAP-based feature explanations
- XGBoost and LightGBM support
- Hyperparameter optimization with Optuna
- Employee clustering
- PDF report generation
- Department-level retention forecasting
- Salary optimization recommendations
- Cloud deployment (AWS/Azure)

---

# 📚 Learning Outcomes

This project demonstrates practical implementation of:

- End-to-End Machine Learning
- Data Preprocessing
- Feature Engineering
- Imbalanced Classification
- Random Forests
- Feature Selection
- Explainable AI
- Business Analytics
- Cost-Benefit Analysis
- Interactive Dashboard Development with Streamlit

---

# 📜 License

This project is intended for educational and portfolio purposes.

---

# 👨‍💻 Author

**Nakul Yawalkar**

If you found this project useful, consider giving it a ⭐ on GitHub!
