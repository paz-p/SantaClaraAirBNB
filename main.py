import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

# ---------------------------------------------------------------------------------------------------------------------
# This loads the data and cleans and prepares it, creating new variables to improve the model's predictive power.

file_path = 'housing.csv'
try:
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
    exit()

housing = df.copy()

# fills missing values and create new engineered features.
median_bedrooms = housing['total_bedrooms'].median()
housing['total_bedrooms'].fillna(median_bedrooms, inplace=True)
housing['rooms_per_household'] = housing['total_rooms'] / housing['households']
housing['bedrooms_per_room'] = housing['total_bedrooms'] / housing['total_rooms']
housing['population_per_household'] = housing['population'] / housing['households']
print("Data preprocessing and feature engineering complete.")

# -----------------------------------------------------------------------------------------------------------------
# This splitting it into training and testing sets, and applies transformations like scaling and one-hot encoding.

X = housing.drop('median_house_value', axis=1)
y = housing['median_house_value']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numerical_features = X.select_dtypes(include=['float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# ColumnTransformer is used to to StandardScaler to numerical data and OneHotEncoder to categorical data so ensuring a clean and reproducible workflow.
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# A pipeline is used to streamline the data transformation process.
data_pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
X_train_prepared = data_pipeline.fit_transform(X_train)
X_test_prepared = data_pipeline.transform(X_test)
print("Data preparation complete!")

# ----------------------------------------------------------------------------------------------------------------------
# This section defines, trains, and evaluate the Stacking Regressor ensemble model to produce prediction.

# Define the base estimators and a final meta-estimator for the stacking ensemble.
estimators = [
    ('rf', RandomForestRegressor(n_estimators=100, max_features=8, random_state=42)),
    ('xgb', xgb.XGBRegressor(n_estimators=100, random_state=42))
]
final_estimator = LinearRegression()

stacking_regressor = StackingRegressor(
    estimators=estimators,
    final_estimator=final_estimator,
    n_jobs=-1
)

print("\nTraining stacking model...")
stacking_regressor.fit(X_train_prepared, y_train)

y_predictions = stacking_regressor.predict(X_test_prepared)

# Evaluate the model's performance using Mean Absolute Error (MAE).
mae = mean_absolute_error(y_test, y_predictions)
print("\nStacking Ensemble Model training complete.")
print(f"Mean Absolute Error (MAE) with Stacking: ${mae:.2f}")

# --------------------------------------------------------------------------------------------------
# This creates a cvs file to import the predtiction to be used on Tableau for better visualization

final_results_df = X_test.copy()
final_results_df['Predicted_Value'] = y_predictions
final_results_df['Actual_Value'] = y_test

# Save the final DataFrame to a CSV file for easy sharing and visualization.
final_results_df.to_csv('final_model_results.csv', index=False)
print("\nFinal results exported to 'final_model_results.csv'")
