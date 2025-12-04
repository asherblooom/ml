import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
# Load the data
file_name = "regression_insurance.csv"
data = pd.read_csv(file_name)
X = data.drop(columns=['charges'])
y = data['charges']

# one hot encode the categorical columns, scale the numerical columns
categorical_cols = ['sex', 'smoker', 'region']
numerical_cols = ['age', 'bmi', 'children']
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(drop='first'), categorical_cols), 
                  ('num', StandardScaler(), numerical_cols)],
    verbose_feature_names_out=False
)

# Split into training and testing sets (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# fit preprocessor on training data, then transform both
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# fit linear regression model
regr = LinearRegression()
regr.fit(X_train_processed, y_train)

# print coefficients
feature_names = preprocessor.get_feature_names_out()
coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': regr.coef_
})
print("Learned Coefficients:")
print(coef_df.round(3))

y_train_pred = regr.predict(X_train_processed)
y_test_pred = regr.predict(X_test_processed)

rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

print("\nRoot Mean Squared Error (RMSE):")
print(f"Training Set: {rmse_train:.3f}")
print(f"Test Set:     {rmse_test:.3f}")

# scatter plot
plt.figure()
plt.scatter(y_test, y_test_pred, alpha=0.5)
# Add a diagonal line for reference
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()])
plt.xlabel('Actual Charges')
plt.ylabel('Predicted Charges')
plt.title('Linear regression: Actual vs. Predicted Charges')
plt.show()