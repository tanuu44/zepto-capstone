import pandas as pd
import numpy as np

df = pd.read_csv("./analytics/titanic.csv")

print("Dataset shape:", df.shape)
print(df.head())

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ============================================================
# MODULE 2 - MODELING PIPELINE
# ============================================================

OUTPUT_DIR = "./analytics/outputs"

# ------------------------------------------------------------
# LOAD THE SAME COMMITTED TITANIC CSV
# ------------------------------------------------------------

df = pd.read_csv("./analytics/titanic.csv")

print("=" * 60)
print("MODULE 2 - MODELING")
print("=" * 60)

print("\nDataset shape:", df.shape)

# ------------------------------------------------------------
# TARGET AND FEATURES
# ------------------------------------------------------------

target = "survived"

# Use independent, non-target features.
# Avoid derived/redundant columns such as:
# alive, adult_male, alone, class, who, embark_town.
features = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "embarked"
]

X = df[features].copy()
y = df[target].copy()

# ------------------------------------------------------------
# TASK 7 - CLASS BALANCE
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("TASK 7 - CLASS BALANCE")
print("=" * 60)

class_counts = y.value_counts().sort_index()
class_percentages = y.value_counts(normalize=True).sort_index() * 100

print("\nClass counts:")
print(class_counts)

print("\nClass percentages:")
print(class_percentages.round(2))

print(
    "\nStratification is used because the Titanic target is imbalanced: "
    "approximately 38.38% survived and 61.62% did not survive. "
    "A stratified split preserves approximately the same class proportions "
    "in both training and test sets."
)

# ------------------------------------------------------------
# STRATIFIED TRAIN/TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)

print("\nTraining class distribution:")
print(y_train.value_counts(normalize=True).sort_index().round(4))

print("\nTest class distribution:")
print(y_test.value_counts(normalize=True).sort_index().round(4))

# ------------------------------------------------------------
# TASK 8 - PREPROCESSING
# ------------------------------------------------------------

numeric_features = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

categorical_features = [
    "sex",
    "embarked"
]

# Numeric preprocessing:
# median imputation + StandardScaler
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

# Categorical preprocessing:
# most-frequent imputation + OneHotEncoder
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_transformer, numeric_features),
        ("categorical", categorical_transformer, categorical_features)
    ]
)

print("\n" + "=" * 60)
print("TASK 8 - PREPROCESSING")
print("=" * 60)

print(
    "\nNumeric columns:",
    numeric_features
)

print(
    "Categorical columns:",
    categorical_features
)

print(
    "\nPreprocessing is fit only on X_train and then applied "
    "to X_test using transform-only behavior through the Pipeline."
)

# Fit only on training data to verify the preprocessing works.
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("\nProcessed training shape:", X_train_processed.shape)
print("Processed test shape:", X_test_processed.shape)

print("\nTASK 7 + TASK 8 COMPLETED")
# Target and features
target = "survived"

features = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "embarked"
]

X = df[features].copy()
y = df[target].copy()

# Task 8: Fit preprocessing ONLY on training data
X_train_processed = preprocessor.fit_transform(X_train)

# Transform test data WITHOUT fitting again
X_test_processed = preprocessor.transform(X_test)
# ============================================================
# TASK 9 - TRAIN THREE CLASSIFIERS
# ============================================================

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

print("=" * 60)
print("TASK 9 - CLASSIFICATION MODELS")
print("=" * 60)

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )
}

fitted_models = {}

for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    pipeline.fit(X_train, y_train)

    fitted_models[name] = pipeline

    print(f"\n{name}")
    print("-" * 40)
    print("Training completed successfully.")

# ============================================================
# DECISION TREE VISUALIZATION
# ============================================================

tree_pipeline = fitted_models["Decision Tree"]

tree_preprocessor = tree_pipeline.named_steps["preprocessor"]
tree_model = tree_pipeline.named_steps["classifier"]

feature_names = tree_preprocessor.get_feature_names_out()

plt.figure(figsize=(24, 12))

plot_tree(
    tree_model,
    feature_names=feature_names,
    class_names=["Not Survived", "Survived"],
    filled=True,
    rounded=True,
    fontsize=8
)

plt.title("Titanic Decision Tree")
plt.tight_layout()

plt.savefig(
    "analytics/outputs/decision_tree.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print("\nDecision tree visualization saved:")
print("analytics/outputs/decision_tree.png")

print("\nTASK 9 COMPLETED")
# ============================================================
# TASK 10 - MODEL EVALUATION
# ============================================================

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score,
    ConfusionMatrixDisplay
)

print("=" * 60)
print("TASK 10 - MODEL EVALUATION")
print("=" * 60)

evaluation_results = []

plt.figure(figsize=(10, 7))

for name, pipeline in fitted_models.items():

    # Predictions
    y_pred = pipeline.predict(X_test)

    # Probability for ROC/AUC
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # Metrics
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    evaluation_results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "AUC": auc
    })

    print(f"\n{name}")
    print("-" * 40)

    print("Confusion Matrix:")
    print(cm)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"AUC      : {auc:.4f}")

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC = {auc:.3f})"
    )

# Random classifier reference line
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - Titanic Classification Models")
plt.legend()
plt.tight_layout()

plt.savefig(
    "analytics/outputs/roc_curves.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# CONFUSION MATRIX CHARTS
# ============================================================

for name, pipeline in fitted_models.items():

    y_pred = pipeline.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Not Survived", "Survived"]
    )

    fig, ax = plt.subplots(figsize=(6, 5))

    disp.plot(ax=ax)

    ax.set_title(f"Confusion Matrix - {name}")

    plt.tight_layout()

    safe_name = name.lower().replace(" ", "_")

    plt.savefig(
        f"analytics/outputs/confusion_matrix_{safe_name}.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

# ============================================================
# COMPARISON TABLE
# ============================================================

results_df = pd.DataFrame(evaluation_results)

print("\n")
print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

results_df.to_csv(
    "analytics/outputs/classification_model_comparison.csv",
    index=False
)

print("\nSaved:")
print("analytics/outputs/classification_model_comparison.csv")
print("analytics/outputs/roc_curves.png")
print("analytics/outputs/confusion_matrix_*.png")

print("\nTASK 10 COMPLETED")
# ============================================================

# TASK 11 - CLASS IMBALANCE

# ============================================================

from imblearn.over_sampling import SMOTE

print("\n" + "=" * 60)
print("TASK 11 - CLASS IMBALANCE")
print("=" * 60)

# ------------------------------------------------------------

# BASELINE RANDOM FOREST

# ------------------------------------------------------------

baseline_rf = RandomForestClassifier(
n_estimators=200,
random_state=42
)

baseline_pipeline = Pipeline([
("preprocessor", preprocessor),
("classifier", baseline_rf)
])

baseline_pipeline.fit(X_train, y_train)

baseline_pred = baseline_pipeline.predict(X_test)

baseline_accuracy = accuracy_score(y_test, baseline_pred)
baseline_precision = precision_score(y_test, baseline_pred)
baseline_recall = recall_score(y_test, baseline_pred)
baseline_f1 = f1_score(y_test, baseline_pred)

print("\nBASELINE RANDOM FOREST")
print("-" * 40)
print(f"Accuracy : {baseline_accuracy:.4f}")
print(f"Precision: {baseline_precision:.4f}")
print(f"Recall   : {baseline_recall:.4f}")
print(f"F1 Score : {baseline_f1:.4f}")

# ------------------------------------------------------------

# CLASS-WEIGHTED RANDOM FOREST

# ------------------------------------------------------------

balanced_rf = RandomForestClassifier(
n_estimators=200,
class_weight="balanced",
random_state=42
)

balanced_pipeline = Pipeline([
("preprocessor", preprocessor),
("classifier", balanced_rf)
])

balanced_pipeline.fit(X_train, y_train)

balanced_pred = balanced_pipeline.predict(X_test)

balanced_accuracy = accuracy_score(y_test, balanced_pred)
balanced_precision = precision_score(y_test, balanced_pred)
balanced_recall = recall_score(y_test, balanced_pred)
balanced_f1 = f1_score(y_test, balanced_pred)

print("\nCLASS-WEIGHTED RANDOM FOREST")
print("-" * 40)
print(f"Accuracy : {balanced_accuracy:.4f}")
print(f"Precision: {balanced_precision:.4f}")
print(f"Recall   : {balanced_recall:.4f}")
print(f"F1 Score : {balanced_f1:.4f}")

# ------------------------------------------------------------

# SMOTE

# ------------------------------------------------------------

print("\nSMOTE")
print("-" * 40)

# IMPORTANT:

# Fit preprocessing ONLY on X_train.

# Never apply SMOTE to X_test.

smote_preprocessor = preprocessor

X_train_processed_smote = smote_preprocessor.fit_transform(X_train)

smote = SMOTE(
random_state=42
)

X_train_smote, y_train_smote = smote.fit_resample(
X_train_processed_smote,
y_train
)

print("Original training class distribution:")
print(y_train.value_counts())

print("\nSMOTE training class distribution:")
print(y_train_smote.value_counts())

print("\nOriginal training shape:", X_train_processed_smote.shape)
print("SMOTE training shape:", X_train_smote.shape)

# ------------------------------------------------------------

# RANDOM FOREST WITH SMOTE

# ------------------------------------------------------------

smote_rf = RandomForestClassifier(
n_estimators=200,
random_state=42
)

smote_rf.fit(
X_train_smote,
y_train_smote
)

# Transform X_test using the already-fitted preprocessor.

X_test_processed_smote = smote_preprocessor.transform(X_test)

smote_pred = smote_rf.predict(
X_test_processed_smote
)

smote_accuracy = accuracy_score(
y_test,
smote_pred
)

smote_precision = precision_score(
y_test,
smote_pred
)

smote_recall = recall_score(
y_test,
smote_pred
)

smote_f1 = f1_score(
y_test,
smote_pred
)

print("\nSMOTE RANDOM FOREST")
print("-" * 40)
print(f"Accuracy : {smote_accuracy:.4f}")
print(f"Precision: {smote_precision:.4f}")
print(f"Recall   : {smote_recall:.4f}")
print(f"F1 Score : {smote_f1:.4f}")

# ------------------------------------------------------------

# CLASS IMBALANCE COMPARISON

# ------------------------------------------------------------

imbalance_results = pd.DataFrame([
{
"Method": "Baseline Random Forest",
"Accuracy": baseline_accuracy,
"Precision": baseline_precision,
"Recall": baseline_recall,
"F1": baseline_f1
},
{
"Method": "Class-Weighted Random Forest",
"Accuracy": balanced_accuracy,
"Precision": balanced_precision,
"Recall": balanced_recall,
"F1": balanced_f1
},
{
"Method": "SMOTE Random Forest",
"Accuracy": smote_accuracy,
"Precision": smote_precision,
"Recall": smote_recall,
"F1": smote_f1
}
])

print("\n" + "=" * 60)
print("CLASS IMBALANCE COMPARISON")
print("=" * 60)

print(
imbalance_results.to_string(
index=False,
float_format=lambda x: f"{x:.4f}"
)
)

imbalance_results.to_csv(
"analytics/outputs/class_imbalance_comparison.csv",
index=False
)

print("\nSaved:")
print("analytics/outputs/class_imbalance_comparison.csv")

print("\nTASK 11 COMPLETED")
# ============================================================
# TASK 12 - HYPERPARAMETER TUNING
# ============================================================

from sklearn.model_selection import GridSearchCV

print("=" * 60)
print("TASK 12 - HYPERPARAMETER TUNING")
print("=" * 60)

# ------------------------------------------------------------
# RANDOM FOREST PIPELINE FOR GRID SEARCH
# ------------------------------------------------------------

tuning_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                random_state=42,
                oob_score=True,
                n_jobs=-1
            )
        )
    ]
)

# ------------------------------------------------------------
# REQUIRED GRID
# ------------------------------------------------------------

param_grid = {
    "classifier__n_estimators": [100, 200, 300],
    "classifier__max_depth": [5, 10, None],
    "classifier__max_features": ["sqrt", "log2"]
}

print("\nGridSearchCV parameter grid:")
print(param_grid)

print("\nScoring metric: F1")
print("Cross-validation: 5-fold")

# ------------------------------------------------------------
# GRID SEARCH
# ------------------------------------------------------------

grid_search = GridSearchCV(
    estimator=tuning_pipeline,
    param_grid=param_grid,
    scoring="f1",
    cv=5,
    n_jobs=-1,
    refit=True
)

print("\nRunning GridSearchCV...")

grid_search.fit(
    X_train,
    y_train
)

print("\nGridSearchCV completed successfully.")

# ------------------------------------------------------------
# BEST PARAMETERS
# ------------------------------------------------------------

print("\nBest parameters:")
print(grid_search.best_params_)

print(
    f"\nBest cross-validation F1: "
    f"{grid_search.best_score_:.4f}"
)

# ------------------------------------------------------------
# BEST TUNED PIPELINE
# ------------------------------------------------------------

tuned_pipeline = grid_search.best_estimator_

# ------------------------------------------------------------
# TEST SET EVALUATION
# ------------------------------------------------------------

y_tuned_pred = tuned_pipeline.predict(X_test)

y_tuned_proba = tuned_pipeline.predict_proba(X_test)[:, 1]

tuned_accuracy = accuracy_score(
    y_test,
    y_tuned_pred
)

tuned_precision = precision_score(
    y_test,
    y_tuned_pred
)

tuned_recall = recall_score(
    y_test,
    y_tuned_pred
)

tuned_f1 = f1_score(
    y_test,
    y_tuned_pred
)

tuned_auc = roc_auc_score(
    y_test,
    y_tuned_proba
)

print("\n" + "=" * 60)
print("TUNED RANDOM FOREST - TEST SET")
print("=" * 60)

print(f"Accuracy : {tuned_accuracy:.4f}")
print(f"Precision: {tuned_precision:.4f}")
print(f"Recall   : {tuned_recall:.4f}")
print(f"F1 Score : {tuned_f1:.4f}")
print(f"AUC      : {tuned_auc:.4f}")

# ------------------------------------------------------------
# OOB VALIDATION
# ------------------------------------------------------------

tuned_rf = tuned_pipeline.named_steps["classifier"]

print("\n" + "=" * 60)
print("OOB VALIDATION")
print("=" * 60)

print(
    f"\nOOB Score: "
    f"{tuned_rf.oob_score_:.4f}"
)

# OOB predictions
oob_decision = tuned_rf.oob_decision_function_

oob_pred = np.argmax(
    oob_decision,
    axis=1
)

oob_proba = oob_decision[:, 1]

oob_accuracy = accuracy_score(
    y_train,
    oob_pred
)

oob_precision = precision_score(
    y_train,
    oob_pred
)

oob_recall = recall_score(
    y_train,
    oob_pred
)

oob_f1 = f1_score(
    y_train,
    oob_pred
)

oob_auc = roc_auc_score(
    y_train,
    oob_proba
)

print("\nOOB Metrics:")
print(f"Accuracy : {oob_accuracy:.4f}")
print(f"Precision: {oob_precision:.4f}")
print(f"Recall   : {oob_recall:.4f}")
print(f"F1 Score : {oob_f1:.4f}")
print(f"AUC      : {oob_auc:.4f}")

# ------------------------------------------------------------
# BASELINE VS TUNED
# ------------------------------------------------------------

baseline_rf = fitted_models["Random Forest"]

baseline_pred = baseline_rf.predict(X_test)

baseline_proba = baseline_rf.predict_proba(X_test)[:, 1]

baseline_accuracy = accuracy_score(
    y_test,
    baseline_pred
)

baseline_precision = precision_score(
    y_test,
    baseline_pred
)

baseline_recall = recall_score(
    y_test,
    baseline_pred
)

baseline_f1 = f1_score(
    y_test,
    baseline_pred
)

baseline_auc = roc_auc_score(
    y_test,
    baseline_proba
)

comparison_task12 = pd.DataFrame([
    {
        "Model": "Baseline Random Forest",
        "Accuracy": baseline_accuracy,
        "Precision": baseline_precision,
        "Recall": baseline_recall,
        "F1": baseline_f1,
        "AUC": baseline_auc
    },
    {
        "Model": "Tuned Random Forest",
        "Accuracy": tuned_accuracy,
        "Precision": tuned_precision,
        "Recall": tuned_recall,
        "F1": tuned_f1,
        "AUC": tuned_auc
    }
])

print("\n" + "=" * 60)
print("BASELINE VS TUNED RANDOM FOREST")
print("=" * 60)

print(
    comparison_task12.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

# ------------------------------------------------------------
# SAVE GRID SEARCH RESULTS
# ------------------------------------------------------------

grid_results = pd.DataFrame(
    grid_search.cv_results_
)

grid_results.to_csv(
    "analytics/outputs/gridsearch_results.csv",
    index=False
)

comparison_task12.to_csv(
    "analytics/outputs/task12_tuned_random_forest_comparison.csv",
    index=False
)

print("\nSaved:")
print("analytics/outputs/gridsearch_results.csv")
print("analytics/outputs/task12_tuned_random_forest_comparison.csv")

print("\nTASK 12 COMPLETED")
# ============================================================
# TASK 13 - REGRESSION SIDE-TASK
# ============================================================

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("=" * 60)
print("TASK 13 - REGRESSION SIDE-TASK")
print("=" * 60)

# ------------------------------------------------------------
# LOAD THE SAME COMMITTED TITANIC DATASET
# ------------------------------------------------------------

reg_df = pd.read_csv("./analytics/titanic.csv")

print("\nRegression dataset shape:", reg_df.shape)

# ------------------------------------------------------------
# TARGET AND FEATURES
# ------------------------------------------------------------

# Predict FARE using the other useful available features.
#
# We exclude:
# - fare -> target
# - alive -> directly derived from survived
# - adult_male -> redundant derived flag
# - alone -> derived from sibsp + parch
# - who -> redundant categorical representation
# - class -> redundant representation of pclass
# - embark_town -> duplicate representation of embarked
# - deck -> very high missingness and not required for this task
#
# This keeps the regression model focused on independent/useful
# predictors rather than redundant variables.

regression_target = "fare"

regression_features = [
    "survived",
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "embarked"
]

X_reg = reg_df[regression_features].copy()
y_reg = reg_df[regression_target].copy()

print("\nRegression features:")
print(regression_features)

print("\nRegression target:")
print(regression_target)

# ------------------------------------------------------------
# TRAIN / TEST SPLIT
# ------------------------------------------------------------

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg,
    y_reg,
    test_size=0.20,
    random_state=42
)

print("\nRegression train shape:", X_reg_train.shape)
print("Regression test shape:", X_reg_test.shape)

# ------------------------------------------------------------
# PREPROCESSING
# ------------------------------------------------------------

reg_numeric_features = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch"
]

reg_categorical_features = [
    "sex",
    "embarked"
]

reg_numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

reg_categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

reg_preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            reg_numeric_transformer,
            reg_numeric_features
        ),
        (
            "categorical",
            reg_categorical_transformer,
            reg_categorical_features
        )
    ]
)

# ------------------------------------------------------------
# LINEAR REGRESSION PIPELINE
# ------------------------------------------------------------

regression_pipeline = Pipeline(
    steps=[
        ("preprocessor", reg_preprocessor),
        ("regressor", LinearRegression())
    ]
)

# IMPORTANT:
# Preprocessing is fitted only on training data.
regression_pipeline.fit(
    X_reg_train,
    y_reg_train
)

# Prediction on untouched test data.
y_reg_pred = regression_pipeline.predict(
    X_reg_test
)

print("\nLinear Regression training completed successfully.")

# ------------------------------------------------------------
# REGRESSION METRICS
# ------------------------------------------------------------

mae = mean_absolute_error(
    y_reg_test,
    y_reg_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_reg_test,
        y_reg_pred
    )
)

r2 = r2_score(
    y_reg_test,
    y_reg_pred
)

# ------------------------------------------------------------
# ADJUSTED R-SQUARED
# ------------------------------------------------------------

n = len(y_reg_test)

# Number of predictors after one-hot encoding.
X_reg_test_processed = regression_pipeline.named_steps[
    "preprocessor"
].transform(X_reg_test)

p = X_reg_test_processed.shape[1]

adjusted_r2 = 1 - (
    (1 - r2) * (n - 1)
    / (n - p - 1)
)

print("\n" + "=" * 60)
print("LINEAR REGRESSION RESULTS")
print("=" * 60)

print(f"MAE         : {mae:.4f}")
print(f"RMSE        : {rmse:.4f}")
print(f"R²          : {r2:.4f}")
print(f"Adjusted R² : {adjusted_r2:.4f}")

print("\nNumber of test observations:", n)
print("Number of predictors after encoding:", p)

# ------------------------------------------------------------
# RESIDUALS
# ------------------------------------------------------------

residuals = y_reg_test - y_reg_pred

# ------------------------------------------------------------
# RESIDUAL PLOT
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.scatter(
    y_reg_pred,
    residuals,
    alpha=0.6
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Predicted Fare")
plt.ylabel("Residuals")
plt.title("Residual Plot - Titanic Fare Linear Regression")

plt.tight_layout()

plt.savefig(
    "analytics/outputs/regression_residual_plot.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print(
    "\nSaved:"
    "\nanalytics/outputs/regression_residual_plot.png"
)

# ------------------------------------------------------------
# HETEROSCEDASTICITY CHECK
# ------------------------------------------------------------

# Compare the average absolute residuals between the
# lower and upper halves of predicted fare values.

residual_check = pd.DataFrame({
    "predicted_fare": y_reg_pred,
    "residual": residuals
})

residual_check["abs_residual"] = (
    residual_check["residual"].abs()
)

median_prediction = residual_check[
    "predicted_fare"
].median()

low_predictions = residual_check[
    residual_check["predicted_fare"] <= median_prediction
]["abs_residual"].mean()

high_predictions = residual_check[
    residual_check["predicted_fare"] > median_prediction
]["abs_residual"].mean()

print("\n" + "=" * 60)
print("HETEROSCEDASTICITY CHECK")
print("=" * 60)

print(
    f"\nMean absolute residual - lower predicted fares: "
    f"{low_predictions:.4f}"
)

print(
    f"Mean absolute residual - higher predicted fares: "
    f"{high_predictions:.4f}"
)

difference_ratio = (
    high_predictions / low_predictions
    if low_predictions != 0
    else np.inf
)

print(
    f"\nResidual spread ratio: "
    f"{difference_ratio:.4f}"
)

if difference_ratio >= 1.5 or difference_ratio <= (1 / 1.5):

    print(
        "\nConclusion: The residual plot suggests possible "
        "heteroscedasticity because the residual spread changes "
        "substantially across predicted fare values."
    )

else:

    print(
        "\nConclusion: The residual plot does not show strong "
        "evidence of heteroscedasticity. The residual spread "
        "appears reasonably consistent across predicted values."
    )

# ------------------------------------------------------------
# SAVE REGRESSION METRICS
# ------------------------------------------------------------

regression_results = pd.DataFrame([
    {
        "Model": "Multivariate Linear Regression",
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Adjusted_R2": adjusted_r2
    }
])

regression_results.to_csv(
    "analytics/outputs/regression_model_metrics.csv",
    index=False
)

print(
    "\nSaved:"
    "\nanalytics/outputs/regression_model_metrics.csv"
)

print("\nTASK 13 COMPLETED")
# ============================================================

# TASK 14 - FINAL MODEL COMPARISON

# ============================================================

print("\n" + "=" * 60)
print("TASK 14 - FINAL MODEL COMPARISON")
print("=" * 60)

# ------------------------------------------------------------

# CLASSIFICATION RESULTS

# ------------------------------------------------------------

final_classification_results = pd.DataFrame([
{
"Model": "Logistic Regression",
"Accuracy": 0.8045,
"Precision": 0.7931,
"Recall": 0.6667,
"F1": 0.7244,
"AUC": 0.8437
},
{
"Model": "Decision Tree",
"Accuracy": 0.7654,
"Precision": 0.7547,
"Recall": 0.5797,
"F1": 0.6557,
"AUC": 0.7971
},
{
"Model": "Random Forest",
"Accuracy": 0.8156,
"Precision": 0.8000,
"Recall": 0.6957,
"F1": 0.7442,
"AUC": 0.8300
}
])

# ------------------------------------------------------------

# REGRESSION RESULTS

# ------------------------------------------------------------

regression_results = {
"MAE": 20.8977,
"RMSE": 30.5328,
"R2": 0.3975,
"Adjusted_R2": 0.3617
}

# ------------------------------------------------------------

# PRINT CLASSIFICATION COMPARISON

# ------------------------------------------------------------

print("\n" + "=" * 60)
print("CLASSIFICATION MODEL COMPARISON")
print("=" * 60)

print(
final_classification_results.to_string(
index=False,
float_format=lambda x: f"{x:.4f}"
)
)

# ------------------------------------------------------------

# PRINT REGRESSION METRICS SEPARATELY

# ------------------------------------------------------------

print("\n" + "=" * 60)
print("REGRESSION MODEL METRICS")
print("=" * 60)

print(f"MAE         : {regression_results['MAE']:.4f}")
print(f"RMSE        : {regression_results['RMSE']:.4f}")
print(f"R²          : {regression_results['R2']:.4f}")
print(
f"Adjusted R² : "
f"{regression_results['Adjusted_R2']:.4f}"
)

# ------------------------------------------------------------

# COMBINED FINAL COMPARISON TABLE

# ------------------------------------------------------------

# Classification and regression metrics are kept in separate

# metric groups and are NOT treated as directly comparable.

final_comparison = final_classification_results.copy()

final_comparison["Regression_MAE"] = regression_results["MAE"]
final_comparison["Regression_RMSE"] = regression_results["RMSE"]
final_comparison["Regression_R2"] = regression_results["R2"]
final_comparison["Regression_Adjusted_R2"] = (
regression_results["Adjusted_R2"]
)

final_comparison.to_csv(
"analytics/outputs/final_model_comparison.csv",
index=False
)

print("\n" + "=" * 60)
print("FINAL MODEL COMPARISON TABLE")
print("=" * 60)

print(
final_comparison.to_string(
index=False,
float_format=lambda x: f"{x:.4f}"
)
)

# ------------------------------------------------------------

# FINAL RECOMMENDATION

# ------------------------------------------------------------

print("\n" + "=" * 60)
print("FINAL RECOMMENDATION")
print("=" * 60)

recommendation = (
"The Random Forest is the recommended classifier for deployment "
"because it achieved the highest test accuracy of 0.8156 and "
"the highest F1 score of 0.7442 among the three baseline "
"classifiers. It also achieved a recall of 0.6957 and precision "
"of 0.8000, providing a good balance between identifying "
"survivors and limiting false-positive predictions. Logistic "
"Regression achieved the highest AUC of 0.8437, slightly above "
"Random Forest's AUC of 0.8300, so it remains a strong alternative "
"when ranking/discrimination is the primary concern. The tuned "
"Random Forest increased precision to 0.8750 and AUC to 0.8431 "
"but reduced recall to 0.6087 and F1 to 0.7179, so the baseline "
"Random Forest provides the better overall balance for deployment."
)

print("\n" + recommendation)

print("\nSaved:")
print("analytics/outputs/final_model_comparison.csv")

print("\nTASK 14 COMPLETED")
# ============================================================
# TASK 15 - SAVE COMPLETE DEPLOYMENT PIPELINE
# ============================================================

import joblib
import os

print("=" * 60)
print("TASK 15 - SAVE COMPLETE PIPELINE")
print("=" * 60)

# ------------------------------------------------------------
# SELECT BEST DEPLOYMENT MODEL
# ------------------------------------------------------------

# Based on Task 14 recommendation:
# Baseline Random Forest provides the best overall balance
# between accuracy, precision, recall and F1.

best_pipeline = fitted_models["Random Forest"]

# ------------------------------------------------------------
# SAVE COMPLETE PIPELINE
# ------------------------------------------------------------

pipeline_path = "analytics/outputs/titanic_survival_pipeline.joblib"

joblib.dump(
    best_pipeline,
    pipeline_path
)

print("\nComplete pipeline saved:")
print(pipeline_path)

# ------------------------------------------------------------
# VERIFY FILE EXISTS
# ------------------------------------------------------------

if os.path.exists(pipeline_path):

    file_size = os.path.getsize(pipeline_path)

    print("\nPipeline file verification:")
    print("File exists: YES")
    print(f"File size: {file_size:,} bytes")

else:

    print("\nERROR: Pipeline file was not created.")

# ------------------------------------------------------------
# RELOAD SAVED PIPELINE
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("RELOADING SAVED PIPELINE")
print("=" * 60)

loaded_pipeline = joblib.load(pipeline_path)

print("\nPipeline successfully reloaded.")

# ------------------------------------------------------------
# VERIFY END-TO-END PREDICTION
# ------------------------------------------------------------

# Raw input data.
# No manual preprocessing is performed here.
# The saved pipeline handles:
# 1. Missing-value imputation
# 2. One-hot encoding
# 3. Standard scaling
# 4. Random Forest prediction

raw_sample = X_test.iloc[:5].copy()

sample_predictions = loaded_pipeline.predict(raw_sample)
sample_probabilities = loaded_pipeline.predict_proba(raw_sample)[:, 1]

print("\nRaw sample input:")
print(raw_sample)

print("\nPredictions:")
print(sample_predictions)

print("\nSurvival probabilities:")
print(sample_probabilities.round(4))

# ------------------------------------------------------------
# VERIFY PREDICTION MATCHES ORIGINAL PIPELINE
# ------------------------------------------------------------

original_predictions = best_pipeline.predict(raw_sample)

if np.array_equal(
    sample_predictions,
    original_predictions
):

    print("\nPrediction verification: PASSED")
    print(
        "Reloaded pipeline produces the same predictions "
        "as the original fitted pipeline."
    )

else:

    print("\nPrediction verification: FAILED")

print("\n" + "=" * 60)
print("TASK 15 COMPLETED")
print("=" * 60)

print("\nSaved artifact:")
print(pipeline_path)

