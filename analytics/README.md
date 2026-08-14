\# Module 2 — Analytics



\## Overview



This module performs exploratory data analysis (EDA), statistical analysis, classification modeling, regression analysis, model evaluation, and final model selection using the Titanic dataset.



The module is divided into two main parts:



\* `01\_eda.py` — Exploratory Data Analysis

\* `02\_modeling.py` — Machine Learning Modeling



\---



\## Project Structure



```text

analytics/

│

├── 01\_eda.py

├── 02\_modeling.py

├── titanic.csv

├── requirements.txt

├── README.md

│

└── outputs/

&#x20;   ├── age\_boxplot.png

&#x20;   ├── age\_histogram.png

&#x20;   ├── fare\_boxplot.png

&#x20;   ├── fare\_histogram.png

&#x20;   ├── chart\_1\_survival\_by\_sex.png

&#x20;   ├── chart\_2\_survival\_by\_class.png

&#x20;   ├── chart\_3\_fare\_by\_class.png

&#x20;   ├── chart\_4\_survival\_class\_sex.png

&#x20;   ├── chart\_5\_age\_fare\_survival.png

&#x20;   ├── correlation\_heatmap.png

&#x20;   ├── decision\_tree.png

&#x20;   ├── roc\_curves.png

&#x20;   ├── confusion\_matrix\_\*.png

&#x20;   ├── cleaned\_titanic.csv

&#x20;   ├── missing\_values.csv

&#x20;   ├── correlation\_matrix.csv

&#x20;   ├── survival\_by\_sex.csv

&#x20;   ├── survival\_by\_pclass.csv

&#x20;   ├── survival\_by\_sex\_pclass.csv

&#x20;   ├── classification\_model\_comparison.csv

&#x20;   ├── class\_imbalance\_comparison.csv

&#x20;   ├── gridsearch\_results.csv

&#x20;   ├── task12\_tuned\_random\_forest\_comparison.csv

&#x20;   ├── regression\_model\_metrics.csv

&#x20;   └── final\_model\_comparison.csv

```



\---



\## Dataset



The analysis uses the Titanic dataset containing 891 passenger records and 15 columns.



Important variables include:



\* `survived`

\* `pclass`

\* `sex`

\* `age`

\* `sibsp`

\* `parch`

\* `fare`

\* `embarked`

\* `embark\_town`

\* `deck`



The target variable for classification is:



```text

survived

```



\---



\## Part A — Exploratory Data Analysis



Run:



```powershell

python .\\01\_eda.py

```



The EDA performs:



1\. Dataset loading and inspection

2\. Shape and data-type analysis

3\. Missing-value analysis

4\. Data cleaning

5\. Age and fare outlier analysis

6\. Fare distribution analysis

7\. Survival analysis by sex

8\. Survival analysis by passenger class

9\. Survival analysis by sex and passenger class

10\. Six-column correlation analysis

11\. Data visualization

12\. Standardization sanity check

13\. Cleaned dataset export



\### Cleaning Decisions



| Column        | Missing | Treatment          |

| ------------- | ------: | ------------------ |

| `age`         |  19.87% | Median imputation  |

| `embarked`    |   0.22% | Drop affected rows |

| `embark\_town` |   0.22% | Drop affected rows |

| `deck`        |  77.22% | Drop column        |



After cleaning, the dataset contains:



```text

889 rows

14 columns

0 missing values

```



\---



\## Part B — Machine Learning Modeling



Run:



```powershell

python .\\02\_modeling.py

```



The modeling pipeline includes:



\### Task 7 — Class Balance



The target distribution is:



```text

Did not survive: 61.62%

Survived:        38.38%

```



A stratified train/test split is used to preserve the class distribution.



\---



\### Task 8 — Preprocessing



Numeric features:



```text

pclasss

age

sibsp

parch

fare

```



Categorical features:



```text

sex

embarked

```



Preprocessing is fitted only on the training data and then applied to the test data through a pipeline.



\---



\### Task 9 — Classification Models



Three baseline classification models are trained:



1\. Logistic Regression

2\. Decision Tree

3\. Random Forest



\---



\### Task 10 — Model Evaluation



The models are evaluated using:



\* Accuracy

\* Precision

\* Recall

\* F1 Score

\* AUC

\* Confusion Matrix

\* ROC Curve



Baseline results:



| Model               | Accuracy | Precision | Recall |     F1 |    AUC |

| ------------------- | -------: | --------: | -----: | -----: | -----: |

| Logistic Regression |   0.8045 |    0.7931 | 0.6667 | 0.7244 | 0.8437 |

| Decision Tree       |   0.7654 |    0.7547 | 0.5797 | 0.6557 | 0.7971 |

| Random Forest       |   0.8156 |    0.8000 | 0.6957 | 0.7442 | 0.8300 |



\---



\### Task 11 — Class Imbalance



Class imbalance was investigated using:



\* Baseline Random Forest

\* Class-Weighted Random Forest

\* SMOTE Random Forest



SMOTE produced:



```text

Original training shape: 712

SMOTE training shape:    878

```



SMOTE Random Forest achieved:



```text

Accuracy : 0.8101

Precision: 0.7612

Recall   : 0.7391

F1 Score : 0.7500

```



\---



\### Task 12 — Hyperparameter Tuning



Random Forest hyperparameters were tuned using `GridSearchCV`.



Parameter grid:



```text

n\_estimators: \[100, 200, 300]

max\_depth: \[5, 10, None]

max\_features: \[sqrt, log2]

```



Cross-validation:



```text

5-fold

```



Scoring metric:



```text

F1

```



Best parameters:



```text

max\_depth = 5

max\_features = sqrt

n\_estimators = 100

```



Best cross-validation F1:



```text

0.7459

```



\---



\### Task 13 — Regression Side Task



A Linear Regression model was trained to predict:



```text

fare

```



Regression metrics:



```text

MAE         : 20.8977

RMSE        : 30.5328

R²          : 0.3975

Adjusted R² : 0.3617

```



The residual analysis suggests possible heteroscedasticity because the residual spread changes substantially across predicted fare values.



\---



\### Task 14 — Final Model Recommendation



The baseline Random Forest is selected as the recommended classifier.



Reason:



\* Highest baseline accuracy: `0.8156`

\* Highest baseline F1: `0.7442`

\* Precision: `0.8000`

\* Recall: `0.6957`



Logistic Regression remains a strong alternative because it achieved the highest baseline AUC:



```text

Logistic Regression AUC: 0.8437

Random Forest AUC:       0.8300

```



The tuned Random Forest improved precision and AUC but reduced recall and F1. Therefore, the baseline Random Forest provides the better overall balance for deployment.



\---



\## Task 15 — Saved Pipeline



The complete preprocessing and modeling pipeline is saved as:



```text

outputs/titanic\_survival\_pipeline.joblib

```



The saved pipeline was successfully reloaded and verified.



The reloaded pipeline produced the same predictions as the original fitted pipeline.



\---



\## How to Run



From the `analytics` directory:



```powershell

python .\\01\_eda.py

python .\\02\_modeling.py

```



Or from the project root:



```powershell

python .\\analytics\\01\_eda.py

python .\\analytics\\02\_modeling.py

```



\---



\## Requirements



Install the analytics dependencies using:



```powershell

pip install -r .\\analytics\\requirements.txt

```



\---



\## Module 2 Status



Module 2 includes:



\* EDA

\* Data cleaning

\* Missing-value analysis

\* Outlier analysis

\* Correlation analysis

\* Visualization

\* Classification

\* Class imbalance analysis

\* SMOTE

\* Hyperparameter tuning

\* Regression

\* Model evaluation

\* Model comparison

\* Pipeline serialization

\* Prediction verification



\*\*Module 2 completed successfully.\*\*



