import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from lasso import LassoRegression
from ridge import RidgeRegression
from gradient_descent import GradientDescent

def rename_columns(data):
    # Rename columns to snake case
    data.columns = data.columns.str.strip().str.replace(r"\s+", " ", regex=True).str.lower().str.replace(" ", "_").str.replace("-", "_").str.replace("/", "_")
    return data

def load_normalize_and_split_data():
    # import data
    life_expectancy_data = pd.read_csv("life_expectancy_data.csv")
    # print(life_expectancy_data)
    life_expectancy_data = rename_columns(life_expectancy_data)
    life_expectancy_data["year"] = life_expectancy_data["year"].astype("str")
    life_expectancy_data.dropna(inplace=True)

    # print(life_expectancy_data)

    target_variable = ["life_expectancy"]
    numerical_variables = ['hiv_aids', 'measles', 'adult_mortality', 'percentage_expenditure', 'alcohol', 'hepatitis_b', 'schooling', 'thinness_1_19_years', 'population', 'gdp', 'under_five_deaths', 'thinness_5_9_years', 'income_composition_of_resources', 'total_expenditure', 'diphtheria', 'infant_deaths', 'polio', 'bmi']
    #set(life_expectancy_data.select_dtypes(["int", "float"]).columns).difference(target_variable)
    # categorical_variables = set(life_expectancy_data.columns).difference(set(life_expectancy_data.select_dtypes(["int", "float"]).columns))

    X = life_expectancy_data[numerical_variables]
    y = life_expectancy_data[target_variable]
    X = (X - X.mean()) / X.std()
    # 1. Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    return X_train, X_test, y_train, y_test

def evaluate_model_performance(model):
    X_train, X_test, y_train, y_test = load_normalize_and_split_data()
    model.fit(X_train,y_train)
    # Make predictions on the test set
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    # Evaluate the model's performance using MSE and R² score
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)

    print(f"Train MSE: {mse_train}")
    print(f"Test MSE: {mse_test}")
    print(f"Train R²: {r2_train}")
    print(f"Test R²: {r2_test}")

def main():

    print("\nLasso with sckit learn: ")
    model = linear_model.Lasso(alpha=0.1)
    evaluate_model_performance(model)

    print("\nRidge with sckit learn: ")
    model = linear_model.Ridge(alpha=0.01)
    evaluate_model_performance(model)

    print("\nLasso without sckit learn: ")
    model = LassoRegression()
    evaluate_model_performance(model)

    print("\nRidge without sckit learn: ")
    model = RidgeRegression()
    evaluate_model_performance(model)

    print("\nRegression without regularization and without sckit learn: ")
    model = GradientDescent()
    evaluate_model_performance(model)

if __name__ == "__main__":
    main()
