# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

###### Step 1: Load the dataset
data = pd.read_csv('../ai_job_market_insights.csv')

# Display first few rows to understand the structure
print(data.head())

# Get summary information
print(data.info())

###### Step 2: Preprocess the data
# Separate the numerical and categorical columns
numerical_cols = ['Salary_USD']
categorical_cols = [col for col in data.columns if col != 'Salary_USD']

# Preprocess data: scale numerical features and encode categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first'), categorical_cols)])

# Apply preprocessing to the data
processed_data = preprocessor.fit_transform(data)

# Convert to DataFrame for easier handling
processed_df = pd.DataFrame(processed_data.toarray(), columns=preprocessor.get_feature_names_out())

print("Preprocessed Data:")
print(processed_df.head())

###### Step 3: Normalize the data (only for numeric columns)
numeric_columns = ['Salary_USD']
scaler = StandardScaler()
data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

###### Step 4: Apply PCA to the preprocessed data
# pca = PCA()
pca = PCA(n_components=3)
pca_data = pca.fit_transform(processed_df)

# Explained variance ratio for each component
explained_variance_ratio = pca.explained_variance_ratio_

# Plot the explained variance (Scree Plot)
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o', linestyle='--')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot')
plt.show()

# Print explained variance ratio for interpretation
print("Explained Variance Ratio by Principal Component:")
for i, var in enumerate(explained_variance_ratio):
    print(f"PC{i+1}: {var:.2%}")

###### Step 5: Cumulative explained variance to determine the optimal number of components
cumulative_variance = pca.explained_variance_ratio_.cumsum()
optimal_components = sum(cumulative_variance <= 0.80) + 1  # Threshold of 80%

# Re-run PCA with the optimal number of components
pca = PCA(n_components=optimal_components)
reduced_data = pca.fit_transform(processed_df)

# Convert reduced data to a DataFrame for easy viewing and analysis
reduced_df = pd.DataFrame(reduced_data, columns=[f'PC{i+1}' for i in range(optimal_components)])

# Display a summary of the reduced dataset
print("Reduced Dataset Summary:")
print(reduced_df.describe())

###### Step 6: Get the loadings (feature contributions to each principal component)
loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(optimal_components)], index=processed_df.columns)

print("Principal Component Loadings:")
print(loadings)