import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

np.random.seed(42)
n = 20

road_analysis = pd.DataFrame({
    'Number_of_Vehicles': np.random.randint(1, 6, n),
    'Speed_limit': np.random.choice([30, 40, 50, 60, 70], n),
    'Weather_Conditions': np.random.choice(['Clear', 'Rain', 'Fog', 'Snow'], n),
    'Road_Surface_Conditions': np.random.choice(['Dry', 'Wet', 'Icy'], n),
    'Light_Conditions': np.random.choice(['Daylight', 'Darkness'], n),
    'Driver_Age': np.random.randint(18, 65, n),
    'Driver_State': np.random.choice(['Normal', 'Tired', 'Under Influence'], n)
})

road_analysis['Accident_Severity'] = (
    0.4 * road_analysis['Number_of_Vehicles'] +
    0.03 * road_analysis['Speed_limit'] +
    np.where(road_analysis['Weather_Conditions'] == 'Rain', 1.5, 0) +
    np.where(road_analysis['Weather_Conditions'] == 'Snow', 2.0, 0) +
    np.where(road_analysis['Road_Surface_Conditions'] == 'Wet', 1.0, 0) +
    np.where(road_analysis['Road_Surface_Conditions'] == 'Icy', 1.8, 0) +
    np.where(road_analysis['Light_Conditions'] == 'Darkness', 1.2, 0) +
    np.where(road_analysis['Driver_State'] == 'Tired', 1.5, 0) +
    np.where(road_analysis['Driver_State'] == 'Under Influence', 2.5, 0) +
    np.where(road_analysis['Driver_Age'] < 25, 1.0, 0) +
    np.random.normal(0, 0.5, n)
)

print(" Road Accident Dataset ")
display(road_analysis)

print("\nSummary statistics (numeric features):")
display(road_analysis.describe())

print("\nCorrelation Matrix (numeric features only):")
numeric_cols = road_analysis.select_dtypes(include=np.number)
corr_matrix = numeric_cols.corr()

plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix of Numeric Features")
plt.show()

categorical_cols = ['Weather_Conditions', 'Road_Surface_Conditions', 'Light_Conditions', 'Driver_State']
for col in categorical_cols:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=col, y='Accident_Severity', data=road_analysis)
    plt.title(f'Accident Severity vs {col}')
    plt.show()

X = road_analysis.drop('Accident_Severity', axis=1)
y = road_analysis['Accident_Severity']

categorical_features = ['Weather_Conditions', 'Road_Surface_Conditions', 'Light_Conditions', 'Driver_State']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ], remainder='passthrough'
)

X_encoded = preprocessor.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nModel Evaluation:\nMean Squared Error: {mse:.2f}\nR^2 Score: {r2:.2f}")

feature_names = preprocessor.get_feature_names_out()
coefficients = model.coef_

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients
}).sort_values(by='Coefficient', ascending=False)

print("\nFeature Importance:")
display(importance_df)

plt.figure(figsize=(10,6))
sns.barplot(x='Coefficient', y='Feature', data=importance_df)
plt.title("Feature Importance on Accident Severity")
plt.show()
