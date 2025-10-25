#preventing diabetes readmission


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

#DATA CLEANING

df = pd.read_excel("diabetic_data_QMH_Club_Fest_2025X.xlsx")
print(f"Original 10 rows: ", df)
print("Before cleaning: ", df.info())

#Replacing ? as null values
df.replace("?", np.nan, inplace = True)

print(df.drop(columns=['body_weight']))
print(f"Missing values: ", df.isnull().sum())
print(df.dropna())
print(f"Duplicate values: ", df.duplicated().sum())
print(df.drop_duplicates(keep="first"))

for col in df.columns:
    if df[col].dtype in ['int64', 'float64']:  # numeric
        df[col].fillna(df[col].mean(), inplace=True)
    else:  # categorical
        df[col].fillna(df[col].mode()[0], inplace=True)

print("\nAfter cleaning: ", df.info())

#Saving cleaned files
df.to_excel("Clean_diabates_data.xlsx", index=False)

print(f"\nData cleaned and saved as: ", 'Clean_diabates_data.xlsx')


#MEASURING CENTRAL TENDENCY AND DISPERSION

df = pd.read_excel("Clean_diabates_data1.xlsx")

print("Columns in dataset:")
print(df.columns.tolist())

selected_cols = [
    'ethnic group', 'sex_identity',
    'age_band', 'body_weight', 'adm_type_code', 'discharge_type',
    'adm_source_data', 'hospital_days', 'insurance_code',
    'provider_specialty', 'lab_test_count', 'procedure_count',
    'medication_count', 'outpatient_visits', 'emergency_visits',
    'inpatient_visits', 'diagnosis_primary', 'diagnosis_secondary',
    'diagnosis_tertiary', 'diagnosis_total', 'glucose_test_result',
    'A1C_result', 'medication_columns', 'med_change_status',
    'diabetic_med_given','readmission_status'
]

df = df[selected_cols]

numeric_df = df.select_dtypes(include=['number'])

# Calculating central tendency
stats_df = pd.DataFrame({
    'Mean': numeric_df.mean(),
    'Median': numeric_df.median(),
    'Std Dev': numeric_df.std()
})

print(stats_df)

plt.figure(figsize=(14, 6))
stats_df.plot(kind='bar', figsize=(14, 6), width=0.8)
plt.title("Mean, Median, and Standard Deviation of Numeric Variables", fontsize=16)
plt.xlabel("Variables", fontsize=12)
plt.ylabel("Value", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title="Statistics", loc='upper right')
plt.tight_layout()
plt.show()


#DETERMINING THE CORRELATION

df_encoded = df.copy()

categorical_cols = df_encoded.select_dtypes(include=['object']).columns

le = LabelEncoder()
for col in categorical_cols:
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

df_encoded['readmission_encoded'] = le.fit_transform(df_encoded['readmission_status'].astype(str))

correlation = df_encoded.corr()['readmission_encoded'].sort_values(ascending=False)
print("\nTop correlations with readmission status:")
print(correlation.head(15))

plt.figure(figsize=(12, 10))
sns.heatmap(
    df_encoded.corr()[['readmission_encoded']].sort_values(by='readmission_encoded', ascending=False),
    annot=True,
    cmap='coolwarm',
    center=0
)
plt.title("Correlation of Variables with Readmission Status", fontsize=14)
plt.show()

top_corr = correlation.head(15)

colors = ['green' if val > 0 else 'red' for val in top_corr.values]
plt.figure(figsize=(10,6))
sns.barplot(x=top_corr.values, y=top_corr.index, palette=colors)
plt.title('Top 15 Variable Correlations with Readmission Status', fontsize=14, weight='bold')
plt.xlabel('Correlation Coefficient')
plt.ylabel('Variables')
plt.show()


#READMISSION STATUS BY DISEASE GROUP

df['readmission_binary'] = df['readmission_status'].replace({
    'NO': 'No',
    '>30': 'Yes',
    '<30': 'Yes'
})

icd_groups = {
    'Circulatory': [(390, 459), (785, 785)],
    'Respiratory': [(460, 519), (786, 786)],
    'Digestive': [(520, 579), (787, 787)],
    'Diabetes': [(250, 251)],
    'Injury': [(800, 999)],
    'Musculoskeletal': [(710, 739)],
    'Genitourinary': [(580, 629), (788, 788)],
    'Neoplasms': [(140, 239)],
    'Other (Symptoms)': [(780, 781), (784, 799)],
    'Endocrine (Excl. DM)': [(240, 279)],
    'Skin/Subcutaneous': [(680, 709), (782, 782)],
    'Infectious': [(1, 139)],
    'Mental': [(290, 319)],
    'External Causes': [],  # E–V codes, skipped as non-numeric
    'Blood Disorders': [(280, 289)],
    'Nervous System': [(320, 359)],
    'Pregnancy/Childbirth': [(630, 679)],
    'Sense Organs': [(360, 389)],
    'Congenital Anomalies': [(740, 759)]
}


def map_icd_group(code):
    try:
        code = float(str(code).split('.')[0])  
    except:
        return 'Unknown'
    for group, ranges in icd_groups.items():
        for low, high in ranges:
            if low <= code < high + 1:
                return group
    return 'Unknown'

df['readmission_status'] = df['readmission_status'].replace({'>30': 'Yes', '<30': 'Yes', 'NO': 'No', 'No': 'No'})

for col in ['diagnosis_primary', 'diagnosis_secondary', 'diagnosis_tertiary']:
    df[col + '_group'] = df[col].apply(map_icd_group)

df['all_diagnoses'] = df[['diagnosis_primary_group', 'diagnosis_secondary_group', 'diagnosis_tertiary_group']].values.tolist()

disease_counts = {}
for _, row in df.iterrows():
    readm = row['readmission_status']
    for group in row['all_diagnoses']:
        if group != 'Unknown':
            disease_counts.setdefault(group, {'Yes': 0, 'No': 0})
            disease_counts[group][readm] += 1


plot_df = pd.DataFrame(disease_counts).T.fillna(0)

plot_df = plot_df.sort_values(by='Yes', ascending=False)

x = np.arange(len(plot_df.index))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - width/2, plot_df['Yes'], width, label='Readmitted (Yes)', color='steelblue')
bars2 = ax.bar(x + width/2, plot_df['No'], width, label='Not Readmitted (No)', color='salmon')

ax.set_xlabel("Disease Group")
ax.set_ylabel("Number of Patients")
ax.set_title("Readmission Status by Disease Group")
ax.set_xticks(x)
ax.set_xticklabels(plot_df.index, rotation=45, ha='right')
ax.legend()

for bars in [bars1, bars2]:
    ax.bar_label(bars, fmt='%d', padding=3, fontsize=8)

plt.tight_layout()
plt.show()


#READMISSION STATUS BY DISEASE GROUP(AMONG DIABETES PATIENTS)

def map_icd_group(code):
    try:
        code = float(str(code).split('.')[0]) 
    except:
        return 'Unknown'
    for group, ranges in icd_groups.items():
        for low, high in ranges:
            if low <= code < high + 1:
                return group
    return 'Unknown'

df = pd.read_excel("Clean_diabates_data1.xlsx")


df['readmission_status'] = df['readmission_status'].replace({
    '>30': 'Yes', '<30': 'Yes', 'NO': 'No', 'No': 'No'
})

for col in ['diagnosis_primary', 'diagnosis_secondary', 'diagnosis_tertiary']:
    df[col + '_group'] = df[col].apply(map_icd_group)

diabetes_patients = df[
    (df['diagnosis_primary_group'] == 'Diabetes') |
    (df['diagnosis_secondary_group'] == 'Diabetes') |
    (df['diagnosis_tertiary_group'] == 'Diabetes')
]

diabetes_patients['all_diagnoses'] = diabetes_patients[
    ['diagnosis_primary_group', 'diagnosis_secondary_group', 'diagnosis_tertiary_group']
].values.tolist()

disease_counts = {}
for _, row in diabetes_patients.iterrows():
    readm = row['readmission_status']
    for group in row['all_diagnoses']:
        if group not in ['Unknown', 'Diabetes']: 
            disease_counts.setdefault(group, {'Yes': 0, 'No': 0})
            disease_counts[group][readm] += 1

plot_df = pd.DataFrame(disease_counts).T.fillna(0)
plot_df = plot_df.sort_values(by='Yes', ascending=False)

x = np.arange(len(plot_df.index))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))

bars1 = ax.bar(x - width/2, plot_df['Yes'], width, label='Readmitted (Yes)', color='#4C72B0')
bars2 = ax.bar(x + width/2, plot_df['No'], width, label='Not Readmitted (No)', color='#DD8452')

ax.set_xlabel("Disease Group", fontsize=12)
ax.set_ylabel("Number of Patients", fontsize=12)
ax.set_title("Readmission Status by Other Diseases (Among Diabetes Patients, Excluding Diabetes)", fontsize=14, pad=15)
ax.set_xticks(x)
ax.set_xticklabels(plot_df.index, rotation=45, ha='right')
ax.legend()
for bars in [bars1, bars2]:
    ax.bar_label(bars, fmt='%d', padding=3, fontsize=8)

plt.tight_layout()
plt.show()


#DISTRIBUTION OF READMISSION STATUS

df['readmission_status'] = df['readmission_status'].replace({
    'NO': 'No',
    'No': 'No',
    '<30': '<30 days',
    '>30': '>30 days'
})

readmission_counts = df['readmission_status'].value_counts()

plt.figure(figsize=(6, 6))
colors = ['#4C72B0', '#55A868', '#C44E52'] 
explode = (0.05, 0.05, 0.05)  

plt.pie(
    readmission_counts,
    labels=readmission_counts.index,
    autopct='%1.1f%%',
    startangle=90,
    colors=colors,
    explode=explode,
    shadow=True,
    textprops={'fontsize': 11}
)

plt.title("Distribution of Readmission Status", fontsize=14, pad=15)
plt.tight_layout()
plt.show()


#DISTRIBUTION OF READMISSION STATUS(ONLY FOR DIABETES PATIENT)

df.columns = df.columns.str.strip().str.lower()
diagnosis_cols = ['diagnosis_primary', 'diagnosis_secondary', 'diagnosis_tertiary']

for col in diagnosis_cols:
    df[col] = df[col].astype(str)

diabetes_patients = df[
    df[diagnosis_cols].apply(
        lambda row: any(code.startswith('250') for code in row), axis=1
    )
]

diabetes_patients['readmission_status'] = diabetes_patients['readmission_status'].replace({
    'NO': 'No',
    'No': 'No',
    '<30': '<30 days',
    '>30': '>30 days'
})

readmission_counts = diabetes_patients['readmission_status'].value_counts()

plt.figure(figsize=(6, 6))
colors = ['#4C72B0', '#55A868', '#C44E52']  
explode = (0.05, 0.05, 0.05)

plt.pie(
    readmission_counts,
    labels=readmission_counts.index,
    autopct='%1.1f%%',
    startangle=90,
    colors=colors,
    explode=explode,
    shadow=True,
    textprops={'fontsize': 11}
)

plt.title("Readmission Status (Only for Diabetes Patients)", fontsize=14, pad=15)
plt.tight_layout()
plt.show()


#BOX-PLOT FOR THE CRUCIAL VARIABLES

df['A1C_result'] = df['A1C_result'].replace({
    '>8': 3,
    '>7': 2,
    'Norm': 1,
    'None': 0
})


df_melted = df.melt(id_vars='readmission_status', 
                    value_vars=['medication_count', 'inpatient_visits', 'outpatient_visits', 'emergency_visits', 'diagnosis_total','A1C_result'],
                    var_name='Variable', 
                    value_name='Value')


plt.figure(figsize=(10,6))
sns.boxplot(x='Variable', y='Value', hue='readmission_status', data=df_melted)

plt.title("Comparison of Multiple Variables vs Readmission Status")
plt.xlabel("Variables")
plt.ylabel("Values")
plt.legend(title='Readmission Status')
plt.show()


#MODEL DESIGN

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


df['readmission_status'] = df['readmission_status'].replace({
    'NO': 0,
    '>30': 1,
    '<30': 1
})
numeric_data = df.select_dtypes(include=['int64', 'float64'])

df['diabetic_med_given'] = df['diabetic_med_given'].map({'Yes': 1, 'No': 0})
df['med_change_status'] = df['med_change_status'].map({'Ch': 1, 'No': 0})

X = df[['inpatient_visits', 'diagnosis_total', 'emergency_visits',
        'outpatient_visits', 'hospital_days',
        'medication_count', 'lab_test_count',
        'procedure_count','med_change_status','diabetic_med_given']]
y = df['readmission_status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

#Model evaluation

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Purples')
plt.title('Confusion Matrix for Readmission Prediction')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

#crosstab analysis

print("\nDifferent factors affecting Readmission status:")
print( pd.crosstab(
        [df['inpatient_visits'],
         df['med_change_status'],
         df['diabetic_med_given'], 
        df['diagnosis_total'],
        df['emergency_visits'],
        df['outpatient_visits'],
        df['hospital_days'],
        df['medication_count'],
        df['lab_test_count'],
        df['procedure_count']],
        df['readmission_status'], 
        normalize='index'
    )
)
