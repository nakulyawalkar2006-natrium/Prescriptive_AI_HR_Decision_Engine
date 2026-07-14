import pandas as pd 
import json

def load_and_localize_data(csv_path,config_path):
    print("Loading data")
    df=pd.read_csv(csv_path)
    with open(config_path,'r') as f:
        config=json.load(f)
    multiplier=config['usd_to_inr_multiplier']
    df['Monthly_Income_INR']=df['MonthlyIncome']*multiplier/4
    df['Annual_CTC_LPA']=(df['Monthly_Income_INR']*12)/100000

    median_salaries=df.groupby('JobRole')['Annual_CTC_LPA'].transform('median')
    df['Comp_Ratio']=df['Annual_CTC_LPA']/median_salaries

    overtime_mapping={'Yes':2.0,'No':1.0}
    df['Overwork_Strain']=df['DistanceFromHome']*df['OverTime'].map(overtime_mapping)

    friction_map=config['role_friction_coefficients']
    df['Role_Friction']=df['JobRole'].map(friction_map).fillna(1.2)
    dropcols=['Over18', 'EmployeeCount', 'StandardHours', 'EmployeeNumber', 'MonthlyIncome', 'Monthly_Income_INR']
    df=df.drop(columns=dropcols)
    return df,config

if __name__ == "__main__":
    df,config=load_and_localize_data('WA_Fn-UseC_-HR-Employee-Attrition.csv','corporate_config.json')
    output_filename='processed_attrition_data.csv'
    df.to_csv(output_filename, index=False)
    print(df[['JobRole', 'Annual_CTC_LPA', 'Comp_Ratio', 'Overwork_Strain', 'Role_Friction']].head(15))