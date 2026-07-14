import pandas as pd
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV,train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score,precision_recall_curve
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectFromModel
import numpy as np
import streamlit as st

@st.cache_resource
def train_attrition_model():
    df=pd.read_csv('processed_attrition_data.csv')
    encoder=LabelEncoder()
    y=encoder.fit_transform(df['Attrition'])
    rawX = df.drop(columns=['Attrition'])
    x=pd.get_dummies(rawX,drop_first=True)

    x_train,x_test,y_train,y_test=train_test_split(x,y,stratify=y,test_size=0.2,random_state=25)

    smote=SMOTE(random_state=25)
    X_train_smote,y_train_smote=smote.fit_resample(x_train,y_train)

    #feature selection
    # Train a quick base model to figure out which columns are statistically useless.
    base_rf=RandomForestClassifier(n_estimators=100,random_state=25)
    base_rf.fit(X_train_smote,y_train_smote)

    # Keep only the features that perform above the 'median' importance level.
    selector=SelectFromModel(base_rf, prefit=True, threshold="median")
    X_train_clean=selector.transform(X_train_smote)
    X_test_clean=selector.transform(x_test)
    kept_features = x_train.columns[selector.get_support()]

    rfc=RandomForestClassifier(n_estimators=800,
                               min_weight_fraction_leaf=0.0,
                               min_samples_split=5,
                               min_samples_leaf=2,
                               min_impurity_decrease=0.001,
                               max_leaf_nodes=None,
                               max_features='sqrt',
                               max_depth=20,
                               criterion='gini',
                               class_weight='balanced',
                               bootstrap=False)


    rfc.fit(X_train_clean,y_train_smote)
    

    pred=rfc.predict(X_test_clean)
    probs=rfc.predict_proba(X_test_clean)[:,1]
    precisions,recalls,thresholds=precision_recall_curve(y_test,probs)
    # Calculate F1 score for every point (adding a tiny number to prevent divide-by-zero errors)
    f1_scores=(2*precisions*recalls)/(precisions+recalls+1e-9)
    # Find the threshold that achieved the highest F1 score
    best_index=np.argmax(f1_scores)
    optimal_threshold = thresholds[best_index]
    new_preds=(probs>=optimal_threshold).astype(int)

    safe_employees=x_train[y_train==0]
    safe_averages=safe_employees.mean()

    return {
        "model": rfc,
        "optimal_threshold": optimal_threshold,
        "kept_features": kept_features,
        "selector": selector,
        "x_test_raw": x_test,       # Needed for your Economic Calculator!
        "X_test_clean": X_test_clean, 
        "y_test": y_test,
        "safe_averages": safe_averages
    }



