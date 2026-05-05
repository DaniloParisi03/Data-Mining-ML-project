import pandas as pd
import numpy as np
import os
from scipy.stats import chi2_contingency
from sklearn.feature_selection import mutual_info_classif

# --- LOADING DATASET ---
print("--- LOADING DATASET ---")
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'online_shoppers_intention.csv')

df = pd.read_csv(file_path)
print(f"Dataset dimensions: {df.shape[0]} rows, {df.shape[1]} columns\n")

# Remove any missing values to avoid calculation errors in Scikit-Learn
df = df.dropna().reset_index(drop=True)

categorical_cols = ['OperatingSystems', 'Browser', 'Region', 'TrafficType', 
                    'VisitorType', 'Weekend', 'Month']

numerical_cols = ['Administrative', 'Administrative_Duration', 'Informational', 
                  'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration', 
                  'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay']


target_col = 'Revenue'

# Create an empty list to store all our results
results_data = []

# ---------------------------------------------------------
# TEST 1: CHI-SQUARE TEST (For Categorical variables)
# (the larger the Χ^2 value, the more likely the variables are related)
# ---------------------------------------------------------
print("Running Chi-Square Test...")

alpha = 0.05

for col in categorical_cols:
    contingency_table = pd.crosstab(df[col], df[target_col])
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    
    conclusion = "KEEP (Correlated)" if p_value < alpha else "DROP (Independent)"
    
    # Append the result as a dictionary
    results_data.append({
        'Feature': col,
        'Feature Type': 'Categorical',
        'Test Applied': 'Chi-Square',
        'Statistic (Chi2 / MI Score)': round(chi2, 4),
        'p-value': round(p_value, 4),
        'Conclusion / Note': conclusion
    })

# ---------------------------------------------------------
# TEST 2: MUTUAL INFORMATION (For Numerical variables)
# ---------------------------------------------------------
print("Running Mutual Information calculation...")
X_num = df[numerical_cols]
y = df[target_col].astype(int)

mi_scores = mutual_info_classif(X_num, y, random_state=42)

for i, col in enumerate(numerical_cols):
    score = mi_scores[i]
    
    # Simple logic for the conclusion note (you can adjust the threshold)
    note = "High Info" if score > 0.05 else "Low Info (Consider Dropping)"
    if score == 0.0:
         note = "Zero Info (DROP)"

    # Append the result
    results_data.append({
        'Feature': col,
        'Feature Type': 'Numerical',
        'Test Applied': 'Mutual Information',
        'Statistic (Chi2 / MI Score)': round(score, 4),
        'p-value': 'N/A', # MI doesn't produce a p-value
        'Conclusion / Note': note
    })

# ---------------------------------------------------------
# EXPORT TO EXCEL
# ---------------------------------------------------------
print("\nSaving results to Excel...")
# Convert the list of dictionaries into a DataFrame
results_df = pd.DataFrame(results_data)

# Sort the DataFrame: first Categorical, then Numerical sorted by MI Score
cat_df = results_df[results_df['Feature Type'] == 'Categorical'].sort_values(by='p-value')
num_df = results_df[results_df['Feature Type'] == 'Numerical'].sort_values(by='Statistic (Chi2 / MI Score)', ascending=False)

final_df = pd.concat([cat_df, num_df])

# Save to Excel in the same directory
output_file = os.path.join(current_dir, 'preprocessing_results.xlsx')
final_df.to_excel(output_file, index=False, sheet_name='Feature Selection')

print(f"V Results successfully saved to: {output_file}")