# Student_Mental_Health_AI

# 🧠 Student Mental Health Score AI

### AI-Powered Student Mental Health Score Prediction & Interactive Analytics Dashboard

An end-to-end Machine Learning project that predicts a student's **Mental Health Score** using demographic, academic, social-media usage, and lifestyle-related information.

The project includes a complete Machine Learning workflow from **data preprocessing and model training to prediction and Streamlit deployment**.

The final application provides an interactive, colorful, and mobile-friendly dashboard where users can enter student information, generate a predicted Mental Health Score, explore dataset patterns, compare Machine Learning models, view feature importance, and download prediction results.

---

## 🚀 Live Application

🔗 **Live Streamlit App:**  
Add your deployed Streamlit URL here.

Example:

https://your-app-name.streamlit.app/

---

## 📌 Project Overview

Student mental health can be influenced by several academic, lifestyle, and digital-behavior factors.

This project uses Machine Learning to estimate a student's **Mental Health Score** based on multiple input features such as:

- Age
- Gender
- Country
- Academic Level
- Most Used Platform
- Purpose of Use
- Average Daily Social Media Usage
- Daily Unlocks
- Study Hours
- Physical Activity Hours
- Sleep Hours
- Stress Level

The application converts these inputs into a predicted numerical **Mental Health Score**.

> ⚠️ This project is intended for educational and demonstration purposes. It is not a medical diagnosis or a replacement for professional mental-health assessment.

---

# 🎯 Project Objectives

The main objectives of this project are:

- Build an end-to-end Machine Learning regression project.
- Perform data preprocessing and feature transformation.
- Train and compare multiple regression algorithms.
- Select the best-performing Machine Learning model.
- Save the trained model using Joblib.
- Build an interactive Streamlit web application.
- Create interactive visualizations using Plotly.
- Provide student-level prediction.
- Create an interactive analytics dashboard.
- Display model performance.
- Display feature importance.
- Maintain prediction history.
- Allow users to download prediction results.
- Make the application responsive and mobile-friendly.
- Deploy the application using Streamlit.

---

# 📊 Dataset

The dataset contains approximately:

- **5,000 student records**
- **13 columns**
- Demographic information
- Academic information
- Social media behavior
- Lifestyle information
- Stress information
- Mental health score

### Dataset Features

| Feature | Description |
|---|---|
| `Age` | Age of the student |
| `Gender` | Gender of the student |
| `Country` | Student's country |
| `Academic_Level` | Academic level |
| `Most_Used_Platform` | Most frequently used social media platform |
| `Purpose_Of_Use` | Main purpose of social media usage |
| `Avg_Daily_Usage_Hours` | Average daily social media usage |
| `Daily_Unlocks` | Number of daily phone/app unlocks |
| `Study_Hours` | Daily study hours |
| `Physical_Activity_Hours` | Daily physical activity |
| `Sleep_Hours_Per_Night` | Average sleep per night |
| `Stress_Level` | Reported stress level |
| `Mental_Health_Score` | Target variable |

---

# 🔄 Machine Learning Workflow

The project follows an end-to-end Machine Learning workflow:

```text
Dataset
   ↓
Data Loading
   ↓
Data Understanding
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Feature Selection
   ↓
Train-Test Split
   ↓
Numerical & Categorical Preprocessing
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Comparison
   ↓
Best Model Selection
   ↓
Model Saving
   ↓
Streamlit Application
   ↓
User Input
   ↓
Prediction
   ↓
Interactive Dashboard
```

---
# 🤖 Machine Learning Models

Six regression algorithms were evaluated:

1. Linear Regression
2. Ridge Regression
3. Decision Tree Regressor
4. Random Forest Regressor
5. Gradient Boosting Regressor
6. Extra Trees Regressor

---

# 🏆 Model Performance

The evaluated models produced the following results:

| Model | R² Score | MAE | RMSE |
|---|---:|---:|---:|
| 🏆 Gradient Boosting | **87.40%** | 0.3597 | 0.4675 |
| Extra Trees | 87.23% | 0.3632 | 0.4706 |
| Random Forest | 86.82% | **0.3572** | 0.4781 |
| Decision Tree | 82.40% | 0.4108 | 0.5524 |
| Linear Regression | 79.34% | 0.4692 | 0.5986 |
| Ridge Regression | 79.31% | 0.4702 | 0.5989 |

## 🥇 Best Model

**Gradient Boosting Regressor**

### Performance

```text
R² Score : 87.40%
MAE      : 0.3597
RMSE     : 0.4675
```

The Gradient Boosting model was selected as the best overall model based on the model evaluation performed in the project.

---

# 🏠  Overview

The Overview section provides:

- Project introduction
- Number of students
- Number of features
- Best R² score
- Best Machine Learning model
- Main project capabilities

### Main Features

```text
🧠 AI Prediction
📊 Interactive Dashboard
📈 Model Evaluation
🔍 Feature Analysis
💡 Smart Insights
📥 Download Results
```

---
# 🛠️ Technologies Used

## Programming Language

```text
Python
```

## Machine Learning

```text
Scikit-learn
```

## Data Analysis

```text
Pandas
NumPy
```

## Visualization

```text
Plotly
```

## Web Application

```text
Streamlit
```

## Model Serialization

```text
Joblib
```

---
---

# 🔮 Future Improvements

Possible future improvements include:

- [ ] Add user authentication
- [ ] Add database support
- [ ] Store prediction history permanently
- [ ] Add downloadable PDF reports
- [ ] Add more interactive charts
- [ ] Add SHAP-based explainability
- [ ] Add model retraining functionality
- [ ] Add automated model monitoring
- [ ] Add additional lifestyle variables
- [ ] Improve mobile navigation
- [ ] Add personalized recommendations
- [ ] Add multiple language support
- [ ] Add cloud database integration

---
# 🎓 Skills Demonstrated

This project demonstrates practical knowledge of:

### Python

- Python programming
- Functions
- Exception handling
- File handling
- Data structures

### Data Analysis

- Pandas
- NumPy
- Data cleaning
- Exploratory Data Analysis
- Statistical analysis

### Machine Learning

- Supervised learning
- Regression
- Train-test split
- Feature preprocessing
- Model comparison
- Model evaluation
- Feature importance
- Model serialization

### Visualization

- Plotly
- Interactive charts
- Dashboard development

### Deployment

- Streamlit
- GitHub
- Requirements management
- Model deployment

---
# 🏆 Key Project Highlights

- ✅ End-to-end Machine Learning project
- ✅ Regression-based prediction
- ✅ 5,000 student records
- ✅ 13 dataset columns
- ✅ Six regression algorithms compared
- ✅ Gradient Boosting selected as best model
- ✅ 87.40% R² score
- ✅ Interactive Streamlit application
- ✅ Plotly visualizations
- ✅ Interactive dashboard
- ✅ Feature importance
- ✅ Prediction history
- ✅ CSV download
- ✅ Responsive mobile UI
- ✅ Joblib model pipeline
- ✅ Streamlit deployment ready
  ---
  # 👨‍💻 Author

**Aditya Wadewale**

# 🏁 Conclusion

This project demonstrates a complete **end-to-end Machine Learning application** for predicting a Student Mental Health Score.

The project started with data analysis and preprocessing, followed by training and evaluation of multiple regression algorithms. After comparing the models, **Gradient Boosting Regressor** was selected as the best-performing model with an **R² score of 87.40%**.

Overall, this project demonstrates how a Machine Learning model can be transformed into a practical **interactive data science application** using Python, Scikit-learn, Plotly, Streamlit, and Joblib.

