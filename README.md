# 🏢 Enterprise HR Retention AI & Workforce Intelligence Platform

An end-to-end **Machine Learning and Decision Support System** that predicts employee attrition, explains the underlying risk factors, evaluates the financial impact of retention strategies, and allows HR teams to simulate interventions before making workforce decisions.

Built with **Python, Scikit-learn, and Streamlit**, the application combines predictive analytics, explainable AI, feature engineering, and business economics into an interactive dashboard for HR professionals.

---

## 🚀 Features

- 🤖 Employee Attrition Prediction
- 👤 Individual Employee Risk Assessment
- 🆕 New Hire Attrition Prediction
- 🧪 Interactive What-If Simulator
- 💰 Retention Cost-Benefit Analysis
- 🧠 Explainable AI
- 📊 Company-wide Flight Risk Dashboard
- 🎯 Random Forest Feature Selection
- ⚖️ Automatic Decision Threshold Optimization
- 📈 Interactive Streamlit Interface

---

# 📸 Dashboard Modules

## 👤 Existing Employee Analysis

Analyze any employee from the dataset and obtain:

- Flight Risk Probability
- Financial Exposure
- Recommended HR Strategy
- Top Risk Factors
- What-If Intervention Simulator

---

## 🆕 New Hire Prediction

Create a completely new employee profile by specifying:

- Age
- Starting Salary
- Years at Company
- Commute Distance
- Business Travel
- Job Satisfaction
- Environment Satisfaction
- Overtime Requirement

The model predicts whether this employee is likely to leave before they are even hired.

---

## 🧪 HR What-If Simulator

Simulate HR interventions such as:

- Salary Increase
- Monthly Compensation Adjustment
- Remove Overtime
- Reduce Overwork
- Improve Job Satisfaction
- Improve Environment Satisfaction
- Improve Job Involvement
- Modify Business Travel Frequency

Every change instantly updates:

- Flight Risk
- Financial Risk
- HR Recommendation

---

## 🗺️ Company Flight Risk Dashboard

Provides organization-wide insights including:

- Average Flight Risk by Job Role
- High-risk Departments
- Attrition Heatmap
- Organization Risk Threshold

---

# 🏗️ System Architecture

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
            Random Forest Classifier
                         │
                         ▼
          Threshold Optimization (F1)
                         │
                         ▼
          Streamlit Decision Dashboard
        ┌────────────────┴────────────────┐
        ▼                                 ▼
 Existing Employee                New Hire Prediction
        ▼                                 ▼
 What-If Simulator             AI Explainability
        ▼                                 ▼
 Financial Analysis        Company Heatmap Analytics
```

---

# 📂 Project Structure

```
.
├── app.py                         # Streamlit dashboard
├── model.py                       # ML model training
├── data_pipeline.py               # Data preprocessing
├── retention_check.py             # Financial decision engine
├── model.ipynb                    # Model development notebook
├── data.ipynb                     # Data exploration notebook
├── processed_attrition_data.csv
├── WA_Fn-UseC_-HR-Employee-Attrition.csv
├── requirements.txt
└── README.md
```

---

# 📊 Machine Learning Pipeline

### Data Preprocessing

The preprocessing stage performs:

- Missing value handling
- Currency conversion
- Annual CTC calculation
- Compensation Ratio generation
- Overwork Strain calculation
- Role Friction calculation
- Feature cleaning

---

### Feature Engineering

Business-specific features include:

- Annual CTC
- Compensation Ratio
- Overwork Strain
- Role Friction

These features provide stronger business context than the original dataset.

---

### Data Preparation

- Label Encoding
- One-Hot Encoding
- Train/Test Split
- SMOTE Oversampling

---

### Feature Selection

A baseline Random Forest identifies statistically important variables.

Only features above the median importance threshold are retained for the final model.

---

### Model

The application uses an optimized **Random Forest Classifier** with:

- Balanced Class Weights
- Feature Bagging
- Optimized Tree Depth
- Optimized Split Parameters
- Bootstrap Disabled

---

### Threshold Optimization

Instead of using the default **0.50 probability threshold**, the application automatically determines the threshold that maximizes the **F1 Score**, improving performance on imbalanced HR datasets.

---

# 🧠 Explainable AI

Rather than acting as a black-box model, the system explains predictions by comparing an employee against the average profile of employees who remained with the company.

For every prediction it highlights:

- Top 5 Risk Factors
- Employee's Current Values
- Healthy Baseline Values
- Largest Deviations

This helps HR understand *why* an employee is considered at risk.

---

# 💰 Economic Decision Engine

The dashboard estimates the financial consequences of employee attrition.

For each employee it calculates:

- Annual Compensation
- Estimated Replacement Cost
- Expected Financial Loss
- Cost of Proposed Raise
- Potential Savings

Recommended strategies include:

- ✅ Stable
- 💵 Give Raise
- 🚪 Allow Exit

This converts machine learning predictions into actionable HR decisions.

---

# 📈 Company Analytics

Organization-wide insights include:

- Average Flight Risk by Job Role
- High-risk Departments
- Flight Risk Visualization
- Attrition Threshold Indicator

These analytics help management identify organizational retention challenges.

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

# 🤖 Machine Learning Techniques

- Random Forest Classifier
- SMOTE Oversampling
- Feature Selection
- Label Encoding
- One-Hot Encoding
- Precision-Recall Curve
- F1 Score Optimization
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

Generate processed dataset:

```bash
python data_pipeline.py
```

Launch the application:

```bash
streamlit run app.py
```

---

# 📋 Workflow

```
Raw HR Dataset
        │
        ▼
Data Preprocessing
        │
        ▼
Feature Engineering
        │
        ▼
Feature Selection
        │
        ▼
Random Forest Training
        │
        ▼
Attrition Prediction
        │
        ▼
Economic Analysis
        │
        ▼
Explainable AI
        │
        ▼
Interactive HR Dashboard
```

---

# 🔮 Future Enhancements

- SHAP Explainability
- XGBoost & LightGBM Models
- Hyperparameter Optimization with Optuna
- Department-level Attrition Forecasting
- PDF Report Generation
- Employee Clustering
- Cloud Deployment (AWS / Azure)
- REST API Support
- Authentication & Multi-user Access
- Real-time HRIS Integration

---

# 📚 Learning Outcomes

This project demonstrates practical implementation of:

- End-to-End Machine Learning
- HR Analytics
- Feature Engineering
- Imbalanced Classification
- Random Forests
- Explainable AI
- Decision Support Systems
- Cost-Benefit Analysis
- Interactive Dashboard Development
- Business Intelligence

---

# 📜 License

This project is intended for educational, research, and portfolio purposes.

---

# 👨‍💻 Author

**Nakul Yawalkar**

If you found this project useful, consider giving it a ⭐ on GitHub!
