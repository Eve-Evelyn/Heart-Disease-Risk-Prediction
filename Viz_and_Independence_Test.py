import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

data = pd.read_csv("framingham.csv")
# print(data.isna().sum())
data = data.dropna()

data['male'] = np.where(data.male == 0, 'female', 'male')
data = data.rename(columns={'male': 'gender'})

fig, axes = plt.subplots(3, 2)
sns.boxplot(data=data, x='TenYearCHD', y="age", ax=axes[0, 0])
sns.boxplot(data=data, x='TenYearCHD', y="totChol", ax=axes[0, 1])
sns.boxplot(data=data, x='TenYearCHD', y="sysBP", ax=axes[1, 0])
sns.boxplot(data=data, x='TenYearCHD', y="diaBP", ax=axes[1, 1])
sns.countplot(data=data, x='diabetes', hue='TenYearCHD', ax=axes[2, 0])
sns.countplot(data=data, x='gender', hue='TenYearCHD', ax=axes[2, 1])
plt.show()

crosstab_CHD_vs_diabetes = pd.crosstab(data['diabetes'],
                                       data['TenYearCHD'],
                                       margins=False)
print(crosstab_CHD_vs_diabetes)
# Result of independence test
result_diabetes = stats.chi2_contingency(crosstab_CHD_vs_diabetes)
# Extract the test result_diabetes
stat = result_diabetes[0]
pval = result_diabetes[1]
print(f"Z-stat diabetes: {stat:.4f}")
print(f"p-value diabetes: {pval:.4f}")

crosstab_CHD_vs_gender = pd.crosstab(data['gender'],
                                     data['TenYearCHD'],
                                     margins=False)
print(crosstab_CHD_vs_gender)
# Result of independence test
result_gender = stats.chi2_contingency(crosstab_CHD_vs_gender)
# Extract the test result_diabetes
stat2 = result_gender[0]
pval2 = result_gender[1]
print(f"Z-stat gender: {stat2:.4f}")
print(f"p-value gender: {pval2:.4f}")
