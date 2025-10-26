
# Medical Insurance Cost Prediction

A data science project that predicts medical insurance costs based on user demographics and health indicators. The model is trained using regression techniques and deployed via FastAPI for real-time inference. Optional Vue 3 frontend included for interactive input.

## Project Overview

This project explores the relationship between medical insurance costs and factors like age, BMI, smoking status, and number of children. It includes:

- Exploratory Data Analysis (EDA)
- Feature engineering and encoding
- Model training and evaluation
- Backend API for predictions
- Optional frontend for user interaction

## Modeling Approach

- Dataset: `insurance.csv`
- Target: `charges`
- Features:
  - Age
  - BMI
  - Children
  - Sex (encoded)
  - Smoker (encoded)
- Models tested:
  - Linear Regression
  - Ridge Regression
  - Random Forest
- Final model saved as `insurance_cost_model.pkl`

# 📈 Evaluation Metrics

- Mean R² Score: Improved from **0.54 → 0.75** through iterative feature engineering and model selection
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

Model performance was validated using cross-validation and multiple regressors including Linear, Ridge, and Random Forest. The final model balances interpretability and accuracy for real-world deployment.

# insurance-predictor
A data-driven project that predicts medical insurance costs using regression modeling. Through iterative feature engineering and model tuning, the mean R² score was improved from 0.54 to 0.75, significantly enhancing predictive accuracy. Built with Python and FastAPI for backend inference, and Vue 3 for optional frontend interaction.
