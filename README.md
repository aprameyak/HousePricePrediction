# House Price Predictor

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white&style=for-the-badge)
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white&style=for-the-badge)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white&style=for-the-badge)

## About

**House Price Predictor** is a machine learning project that leverages linear regression to predict housing prices. The application reads housing data from a CSV file, performs data cleaning and preprocessing, and then builds a predictive model using scikit-learn. This project demonstrates how to handle categorical variables, scale features, and evaluate model performance using the R² score.

## Features

- **Data Loading & Cleaning:** Reads a housing dataset and cleans binary categorical fields (e.g., mainroad, guestroom) by mapping yes/no values to numerical ones.
- **Feature Engineering:** Converts furnishing status into a numerical value and drops non-relevant columns.
- **Model Building:** Splits data into training and testing sets, applies standard scaling, and builds a linear regression model.
- **Evaluation:** Evaluates model performance using the R² score.
- **Reproducible Workflow:** Provides a clear, reproducible pipeline for data preprocessing and modeling.

## Technology Stack

- **Language:** Python  
- **Data Processing:** Pandas, NumPy  
- **Machine Learning:** scikit-learn  
- **Visualization:** Matplotlib (if needed for further EDA)  
