import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


df = pd.read_csv("heart-disease.csv")


mean_age = df['Age'].mean()
print(f"a) Mean Age: {mean_age:.2f}")


heart_counts = df['HeartDisease'].value_counts()
print("\nb) Heart Disease Counts:")
print(heart_counts)


max_chol = df['Cholesterol'].max()
min_chol = df['Cholesterol'].min()
print(f"\nc) Max Cholesterol: {max_chol}")
print(f"   Min Cholesterol: {min_chol}")

# ===============================
# d) BMI > 30
# ===============================
bmi_over_30 = (df['BMI'] > 30).sum()
print(f"\nd) Individuals with BMI > 30: {bmi_over_30}")

# ===============================
# e) High BP & Heart Disease
# ===============================
highbp_hd = df[(df['HighBP'] == 1) & (df['HeartDisease'] == 1)].shape[0]
print(f"\ne) High BP & Heart Disease: {highbp_hd}")

# ===============================
# BMI Distribution
# ===============================
plt.figure(figsize=(8, 5))
sns.histplot(df['BMI'], bins=20)
plt.title("BMI Distribution")
plt.xlabel("BMI")
plt.ylabel("Count")
plt.show()

# ===============================
# f) Most common BMI range
# ===============================
bmi_counts, bins = np.histogram(df['BMI'], bins=20)
most_common_bin = (bins[bmi_counts.argmax()], bins[bmi_counts.argmax() + 1])
print(f"\nf) Most common BMI range: {most_common_bin}")

# ===============================
# Exercise Habits vs Heart Disease
# ===============================
plt.figure(figsize=(8, 5))
sns.countplot(x='ExerciseHabits', hue='HeartDisease', data=df)
plt.title("Heart Disease by Exercise Habits")
plt.xlabel("Exercise Habits")
plt.ylabel("Count")
plt.show()

# ===============================
# h) Lowest risk group
# ===============================
risk_by_exercise = df.groupby('ExerciseHabits')['HeartDisease'].mean()
lowest_risk_group = risk_by_exercise.idxmin()
print(f"\nh) Lowest risk group: {lowest_risk_group}")

# ===============================
# Smokers vs Non-Smokers
# ===============================
smoker_counts = df['Smoker'].value_counts()

plt.figure(figsize=(6, 6))
plt.pie(
    smoker_counts,
    labels=['Non-Smoker', 'Smoker'],
    autopct='%1.1f%%'
)
plt.title("Smokers vs Non-Smokers")
plt.show()

# ===============================
# j) Percentage of smokers
# ===============================
smoker_percentage = (df['Smoker'].sum() / len(df)) * 100
print(f"\nj) Percentage of smokers: {smoker_percentage:.2f}%")

# ===============================
# BMI vs Triglycerides
# ===============================
plt.figure(figsize=(8, 5))
sns.scatterplot(
    x='BMI',
    y='Triglyceride',
    hue='HeartDisease',
    data=df
)
plt.title("BMI vs Triglyceride Level")
plt.xlabel("BMI")
plt.ylabel("Triglyceride")
plt.show()

print(
    "\nl) Scatter plot se trend dekhein "
    "(jaise zyada BMI ke sath zyada triglycerides)."
)

# ===============================
# Sleep Hours Boxplot
# ===============================
plt.figure(figsize=(6, 5))
sns.boxplot(y='SleepHours', data=df)
plt.title("Sleep Hours Distribution")
plt.ylabel("Sleep Hours")
plt.show()

# ===============================
# n) Median Sleep Hours
# ===============================
median_sleep = df['SleepHours'].median()
print(f"\nn) Median Sleep Hours: {median_sleep}")

# ===============================
# o) Outliers using IQR
# ===============================
q1 = df['SleepHours'].quantile(0.25)
q3 = df['SleepHours'].quantile(0.75)
iqr = q3 - q1

outliers = df[
    (df['SleepHours'] < (q1 - 1.5 * iqr)) |
    (df['SleepHours'] > (q3 + 1.5 * iqr))
]

print(f"\no) Number of outliers in Sleep Hours: {outliers.shape[0]}")

