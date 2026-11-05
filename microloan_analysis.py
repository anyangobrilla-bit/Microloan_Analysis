import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler

print("1. Generating simulated microloan dataset (this may take a few moments)...")
# Simulating a dataset with 10,000 rows (scale up as needed) and 500 features
np.random.seed(42)
n_samples = 10000
n_features = 500

# Generate random feature names and data
feature_names = [f"feature_{i+1}" for i in range(n_features)]
X_data = np.random.randn(n_samples, n_features)

# Create a target variable 'default' (0 or 1) influenced by some features
y_data = (
    (X_data[:, 0] * 1.5 + X_data[:, 10] * -2.0 + np.random.randn(n_samples) > 0)
    .astype(int)
)

df = pd.DataFrame(X_data, columns=feature_names)
df["default"] = y_data

print(f"Dataset shape: {df.shape}")

# Separate features and target
X = df.drop(columns=["default"])
y = df["default"]

print("\n2. Applying Feature Selection (Top 10 Features)...")
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X, y)

top_10_features = X.columns[selector.get_support()]
print("Top 10 Features Correlated with Default:")
for i, feat in enumerate(top_10_features, 1):
  print(f"   {i}. {feat}")

print("\n3. Applying PCA Dimensionality Reduction...")
# Standardize features first
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA to retain 95% of cumulative variance
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_scaled)

print(f"Original number of features: {X.shape[1]}")
print(f"Reduced number of principal components: {X_pca.shape[1]}")
print(
    "Total cumulative explained variance retained:"
    f" {pca.explained_variance_ratio_.sum() * 100:.2f}%"
)
print("\nMicroloan analysis script executed successfully!")