# 🧠 Student Mental Health Score Prediction

A Machine Learning regression project that predicts a student's **Mental Health Score** using social media usage, study habits, sleep, physical activity, stress level, and demographic information.

The project covers the complete ML workflow — from **data exploration and cleaning to preprocessing, model training, hyperparameter tuning, evaluation, and model saving**.

---

## 📌 Project Overview

Mental health can be influenced by several lifestyle and behavioral factors. This project explores whether patterns in **social media usage, study time, sleep, physical activity, and stress** can be used to predict a student's mental health score.

Since `Mental_Health_Score` is a continuous numerical value, this is treated as a **regression problem**.

### 🎯 Objective

Build a machine learning model that can predict:

```text
Mental_Health_Score
```

using information such as:

* Social media usage
* Daily phone unlocks
* Study hours
* Sleep hours
* Physical activity
* Stress level
* Gender
* Academic level
* Country
* Most-used platform
* Purpose of social media usage

---

## 📊 Dataset

**Dataset:** `Student_Social_Media_And_Mental_Health_Impact.csv`

The dataset contains approximately **5,000 students** and **13 original columns** covering demographics, social media behavior, lifestyle, and stress-related information.

### Important Features

| Feature                   | Description                          |
| ------------------------- | ------------------------------------ |
| `Age`                     | Student's age                        |
| `Gender`                  | Student's gender                     |
| `Country`                 | Student's country                    |
| `Avg_Daily_Usage_Hours`   | Average daily social media usage     |
| `Daily_Unlocks`           | Number of daily device unlocks       |
| `Most_Used_Platform`      | Most frequently used social platform |
| `Study_Hours`             | Daily study hours                    |
| `Sleep_Hours_Per_Night`   | Average nightly sleep                |
| `Physical_Activity_Hours` | Physical activity hours              |
| `Stress_Level`            | Low, Medium, High, or Very High      |
| `Mental_Health_Score`     | Target variable                      |

---

## 🛠️ Technologies & Libraries

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib
* Jupyter Notebook / Google Colab

---

## 🔄 Machine Learning Workflow

```text
Dataset
   ↓
Data Loading
   ↓
Data Exploration
   ↓
EDA & Visualization
   ↓
Data Cleaning
   ↓
Feature Engineering
   ↓
Encoding & Scaling
   ↓
Train/Test Split
   ↓
Preprocessing Pipeline
   ↓
Model Training
   ↓
Hyperparameter Tuning
   ↓
Model Evaluation
   ↓
Save Model
   ↓
FastAPI + Frontend (Next Step)
```

---

## 🔍 Exploratory Data Analysis

The project explores several important relationships in the dataset.

### Visualizations include:

* Mental health score distribution
* Correlation heatmap
* Stress level vs. mental health score
* Social media usage vs. mental health score
* Sleep hours vs. mental health score
* Most-used social media platforms

These visualizations help understand patterns in the data before training the models.

---

## 🧹 Data Cleaning

The dataset was checked for:

* Missing values
* Duplicate rows
* Invalid values
* Outliers
* Skewness

One invalid value was identified in `Physical_Activity_Hours`, where a negative number of hours appeared.

Instead of deleting the entire row, the value was clipped to a minimum of `0`:

```python
df['Physical_Activity_Hours'] = df['Physical_Activity_Hours'].clip(lower=0)
```

This preserves the rest of the student's valid information.

---

## ⚙️ Feature Engineering

### Country Grouping

The dataset contains **111 unique countries**.

Instead of creating a large number of one-hot encoded columns, the project keeps the **10 most frequent countries** and groups the remaining countries into:

```text
Other
```

This reduces unnecessary dimensionality while retaining useful information.

---


## 🔢 Encoding Strategy

Different types of categorical features require different encoding methods.

### Ordinal Encoding

`Stress_Level` has a natural order:

```text
Low → Medium → High → Very High
```

Therefore, it is encoded using ordinal encoding:

```text
Low        → 0
Medium     → 1
High       → 2
Very High  → 3
```

### One-Hot Encoding

Features without a natural ranking are one-hot encoded:

* Gender
* Academic Level
* Most Used Platform
* Purpose of Use
* Grouped Country

---

## 🧪 Train-Test Split

The dataset is divided into:

```text
Training Data → 70%
Testing Data  → 30%
```

A separate test set allows the models to be evaluated on data they have not seen during training.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)
```

---

## 🔧 Preprocessing Pipeline

A `ColumnTransformer` is used to apply different preprocessing operations to different feature types.

### Numerical Features

* Scaling using `StandardScaler`

### Skewed Feature

`Study_Hours` is transformed using:

```python
np.log1p()
```

and then scaled.

### Stress Level

Ordinal encoding with an explicitly defined order.

### Categorical Features

One-hot encoding with:

```python
OneHotEncoder(handle_unknown="ignore")
```

---

## 🤖 Machine Learning Models

Two regression approaches are tested.

### 1. Linear Regression

Used as the baseline model.

```python
LinearRegression()
```

### 2. Random Forest Regressor

A tree-based model capable of learning non-linear relationships.

```python
RandomForestRegressor(random_state=42)
```

---

## 🎯 Hyperparameter Tuning

The Random Forest model is further optimized using:

```python
RandomizedSearchCV
```

Parameters explored include:

* `n_estimators`
* `max_depth`
* `min_samples_split`
* `min_samples_leaf`

Five-fold cross-validation is used during the search.

---

## 📈 Model Evaluation

The models are evaluated using three regression metrics.

### R² Score

Measures how much variation in the target variable is explained by the model.

Higher values indicate that the model explains more of the observed variation.

### MAE — Mean Absolute Error

Measures the average absolute difference between predicted and actual scores.

```text
Lower MAE = smaller average prediction error
```

### RMSE — Root Mean Squared Error

Similar to MAE, but gives more weight to larger prediction errors.

```text
Lower RMSE = fewer/lower large prediction errors
```

The project compares:

| Model               | R² | MAE | RMSE |
| ------------------- | -: | --: | ---: |
| Linear Regression   |  — |   — |    — |
| Random Forest       |  — |   — |    — |
| Tuned Random Forest |  — |   — |    — |

> The values are generated when the notebook is executed and are intentionally not hard-coded here.

---

## 💾 Model Saving

The complete Random Forest pipeline is saved using Joblib:

```python
joblib.dump(
    rf_pipeline,
    'Mental_Health_Model.pkl'
)
```

The saved pipeline contains both:

```text
Preprocessing + Model
```

This means new raw input can be passed through the same preprocessing steps automatically during prediction.

---

## 🚀 Future Development

The next stage of the project is to turn the trained model into a usable application.

### Planned Architecture

```text
User
 ↓
Frontend
 ↓
FastAPI
 ↓
Saved ML Pipeline
 ↓
Prediction
 ↓
Mental Health Score
```

### Planned Features

* FastAPI backend
* Pydantic input validation
* `/predict` API endpoint
* Simple frontend
* Model deployment
* Interactive prediction interface

---

## 📁 Project Structure

```text
student-mental-health-prediction/
│
├── data/
│   └── Student_Social_Media_And_Mental_Health_Impact.csv
│
├── notebooks/
│   └── mental_health_prediction.ipynb
│
├── models/
│   └── Mental_Health_Model.pkl
│
├── README.md
└── requirements.txt
```

---

## ⚠️ Disclaimer

This project is intended for **educational and machine-learning experimentation purposes**.

The predicted score should **not** be treated as a medical diagnosis or professional mental-health assessment.

---

## 👩‍💻 Author

**Syeda Sakina Shahzadi Bukhari **

Computer Science Student
Machine Learning & AI Enthusiast

---

## ⭐ Project Status

**Current:** Machine Learning model development completed

**Next:** FastAPI backend → Frontend → Deployment
opment completed

Next: FastAPI backend → Frontend → Deployment
