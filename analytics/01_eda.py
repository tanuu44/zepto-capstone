import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ============================================================
# MODULE 2 - PART A: EDA
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "titanic.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("MODULE 2 - TITANIC EDA")
print("=" * 60)

# ------------------------------------------------------------
# 1. LOAD OFFLINE FALLBACK
# ------------------------------------------------------------
df = pd.read_csv(CSV_PATH)

print("\nDATASET LOADED")
print("-" * 60)

print("\nShape:")
print(df.shape)

print("\nInfo:")
df.info()

print("\nDescribe:")
print(df.describe())

# Missing values
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)

missing_report = pd.DataFrame({
    "Missing": missing,
    "Percentage": missing_pct
})

print("\nMissing values:")
print(missing_report[missing > 0])

# Save missing report
missing_report[missing > 0].to_csv(
    os.path.join(OUTPUT_DIR, "missing_values.csv")
)

# ------------------------------------------------------------
# 2. CLEANING
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("CLEANING")
print("=" * 60)

cleaned_df = df.copy()

# age = 19.87% -> impute median
cleaned_df["age"] = cleaned_df["age"].fillna(
    cleaned_df["age"].median()
)

# embarked = 0.22% -> drop affected rows
cleaned_df = cleaned_df.dropna(subset=["embarked"])

# embark_town = 0.22% -> drop affected rows
cleaned_df = cleaned_df.dropna(subset=["embark_town"])

# deck = 77.22% -> too much missing, drop column
cleaned_df = cleaned_df.drop(columns=["deck"])

print("\nCleaning decisions:")
print("age: 19.87% missing -> median imputation")
print("embarked: 0.22% missing -> drop rows")
print("embark_town: 0.22% missing -> drop rows")
print("deck: 77.22% missing -> drop column")

print("\nRemaining missing values:")
print(cleaned_df.isnull().sum().sum())

# ------------------------------------------------------------
# 3. AGE AND FARE OUTLIERS
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("OUTLIER ANALYSIS")
print("=" * 60)


def iqr_outliers(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    count = ((series < lower) | (series > upper)).sum()

    return q1, q3, iqr, lower, upper, count


for column in ["age", "fare"]:
    q1, q3, iqr, lower, upper, count = iqr_outliers(
        cleaned_df[column]
    )

    print(f"\n{column.upper()}")
    print(f"Q1: {q1:.2f}")
    print(f"Q3: {q3:.2f}")
    print(f"IQR: {iqr:.2f}")
    print(f"Lower bound: {lower:.2f}")
    print(f"Upper bound: {upper:.2f}")
    print(f"IQR outliers: {count}")

# Fare statistics
fare_mean = cleaned_df["fare"].mean()
fare_median = cleaned_df["fare"].median()
fare_mode = cleaned_df["fare"].mode().iloc[0]

print("\nFARE STATISTICS")
print(f"Mean: {fare_mean:.2f}")
print(f"Median: {fare_median:.2f}")
print(f"Mode: {fare_mode:.2f}")

if fare_mean > fare_median > fare_mode:
    skew_result = "Right-skewed"
elif fare_mean < fare_median < fare_mode:
    skew_result = "Left-skewed"
else:
    skew_result = "Not strictly determined by mean/median/mode ordering"

print(f"Distribution: {skew_result}")

# ------------------------------------------------------------
# 4. UNIVARIATE CHARTS
# ------------------------------------------------------------

# Age histogram
plt.figure(figsize=(8, 5))
sns.histplot(cleaned_df["age"], kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "age_histogram.png"))
plt.close()

# Age boxplot
plt.figure(figsize=(8, 5))
sns.boxplot(x=cleaned_df["age"])
plt.title("Age Box Plot")
plt.xlabel("Age")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "age_boxplot.png"))
plt.close()

# Fare histogram
plt.figure(figsize=(8, 5))
sns.histplot(cleaned_df["fare"], kde=True)
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fare_histogram.png"))
plt.close()

# Fare boxplot
plt.figure(figsize=(8, 5))
sns.boxplot(x=cleaned_df["fare"])
plt.title("Fare Box Plot")
plt.xlabel("Fare")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fare_boxplot.png"))
plt.close()

# ------------------------------------------------------------
# 5. BIVARIATE SURVIVAL ANALYSIS
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("SURVIVAL ANALYSIS")
print("=" * 60)

sex_survival = cleaned_df.groupby("sex")["survived"].mean()

pclass_survival = cleaned_df.groupby("pclass")["survived"].mean()

sex_pclass_survival = (
    cleaned_df
    .groupby(["sex", "pclass"])["survived"]
    .mean()
)

print("\nSurvival rate by sex:")
print(sex_survival)

print("\nSurvival rate by pclass:")
print(pclass_survival)

print("\nSurvival rate by sex and pclass:")
print(sex_pclass_survival)

sex_survival.to_csv(
    os.path.join(OUTPUT_DIR, "survival_by_sex.csv")
)

pclass_survival.to_csv(
    os.path.join(OUTPUT_DIR, "survival_by_pclass.csv")
)

sex_pclass_survival.to_csv(
    os.path.join(OUTPUT_DIR, "survival_by_sex_pclass.csv")
)

# ------------------------------------------------------------
# 6. CORRELATION MATRIX - EXACT SIX COLUMNS
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("CORRELATION MATRIX")
print("=" * 60)

corr_columns = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

corr_matrix = cleaned_df[corr_columns].corr()

print(corr_matrix)

corr_matrix.to_csv(
    os.path.join(OUTPUT_DIR, "correlation_matrix.csv")
)

plt.figure(figsize=(9, 7))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title("Titanic Numeric Feature Correlation")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "correlation_heatmap.png")
)
plt.close()

# Find two strongest correlations
pairs = []

for i in range(len(corr_columns)):
    for j in range(i + 1, len(corr_columns)):
        col1 = corr_columns[i]
        col2 = corr_columns[j]
        value = corr_matrix.loc[col1, col2]

        pairs.append(
            (col1, col2, value, abs(value))
        )

pairs = sorted(
    pairs,
    key=lambda x: x[3],
    reverse=True
)

print("\nTwo strongest correlations:")

for pair in pairs[:2]:
    print(
        f"{pair[0]} vs {pair[1]}: "
        f"{pair[2]:.4f}"
    )

# ------------------------------------------------------------
# 7. MULTIVARIATE DATA STORY CHARTS
# ------------------------------------------------------------

# Chart 1
plt.figure(figsize=(8, 5))
sns.barplot(
    data=cleaned_df,
    x="sex",
    y="survived"
)
plt.title("Survival Rate by Sex")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "chart_1_survival_by_sex.png")
)
plt.close()

# Chart 2
plt.figure(figsize=(8, 5))
sns.barplot(
    data=cleaned_df,
    x="pclass",
    y="survived"
)
plt.title("Survival Rate by Passenger Class")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "chart_2_survival_by_class.png")
)
plt.close()

# Chart 3
plt.figure(figsize=(8, 5))
sns.boxplot(
    data=cleaned_df,
    x="pclass",
    y="fare"
)
plt.title("Fare Distribution by Passenger Class")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "chart_3_fare_by_class.png")
)
plt.close()

# Chart 4
plt.figure(figsize=(8, 5))
sns.barplot(
    data=cleaned_df,
    x="pclass",
    y="survived",
    hue="sex"
)
plt.title("Survival by Class and Sex")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "chart_4_survival_class_sex.png")
)
plt.close()

# Chart 5
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=cleaned_df,
    x="age",
    y="fare",
    hue="survived",
    alpha=0.6
)
plt.title("Age vs Fare by Survival")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "chart_5_age_fare_survival.png")
)
plt.close()

# ------------------------------------------------------------
# 8. STANDARDIZATION SANITY CHECK
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("STANDARDIZATION CHECK")
print("=" * 60)

before = cleaned_df[["age", "fare"]].agg(
    ["mean", "std"]
)

standardized = cleaned_df[["age", "fare"]].copy()

standardized["age"] = (
    standardized["age"] - standardized["age"].mean()
) / standardized["age"].std()

standardized["fare"] = (
    standardized["fare"] - standardized["fare"].mean()
) / standardized["fare"].std()

after = standardized.agg(["mean", "std"])

print("\nBEFORE:")
print(before)

print("\nAFTER:")
print(after)

after.to_csv(
    os.path.join(OUTPUT_DIR, "standardization_after.csv")
)

# ------------------------------------------------------------
# 9. SAVE CLEANED DATA
# ------------------------------------------------------------
cleaned_df.to_csv(
    os.path.join(OUTPUT_DIR, "cleaned_titanic.csv"),
    index=False
)

print("\n" + "=" * 60)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 60)

print(f"\nCleaned rows: {len(cleaned_df)}")
print(f"Cleaned columns: {len(cleaned_df.columns)}")
print(f"Output folder: {OUTPUT_DIR}")