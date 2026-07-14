import pandas as pd
import numpy as np
def analyze_retention(row_idx,risk_prob,test_df,raise_precent=0.15):
    emp_data=test_df.iloc[row_idx]
    annual_ctc=emp_data['Annual_CTC_LPA']

    replacement_cost=annual_ctc*0.6
    expected_loss=risk_prob*replacement_cost
    cost_of_raise=annual_ctc*raise_precent

    if risk_prob < 0.30:
        decision="Stable"
        details="Flight risk is low. No immediate salary adjustments recommended."
        savings=0.0
    elif cost_of_raise<expected_loss:
        decision="Give Raise"
        details=f"Economically beneficial to retain. Offer a {raise_precent*100:.0f}% counter-raise."
        savings=expected_loss-cost_of_raise
    else:
        decision="Allow Exit"
        details="The cost of the requested raise exceeds the statistical financial risk of replacement."
        savings=cost_of_raise-expected_loss

    return {
        "employee_id": f"Employee Row #{row_idx}",
        "current_ctc": float(annual_ctc),
        "flight_risk": float(risk_prob),
        "replacement_cost": float(replacement_cost),
        "risk_adjusted_loss": float(expected_loss),
        "cost_of_raise": float(cost_of_raise),
        "recommended_action": decision,
        "strategic_details": details,
        "potential_savings": float(savings)
    }
    