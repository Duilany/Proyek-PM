import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import json

# Load data
data = pd.read_csv('Data_Tanaman_Padi_Sumatera_version_1.csv')

# Step 1: Clean outliers with stricter threshold
Q1 = data['Produksi'].quantile(0.25)
Q3 = data['Produksi'].quantile(0.75)
IQR = Q3 - Q1
data = data[~((data['Produksi'] < (Q1 - 3.0 * IQR)) | (data['Produksi'] > (Q3 + 3.0 * IQR)))]

# Additional anomaly cleaning
data = data[
    (data['Curah hujan'].between(500, 5000)) &
    (data['Kelembapan'].between(60, 90)) &
    (data['Suhu rata-rata'].between(25, 32))
]

# Step 2: Feature engineering
data['Rata_Produksi_Provinsi'] = data.groupby('Provinsi')['Produksi'].transform('mean')
data['Tahun_Norm'] = (data['Tahun'] - data['Tahun'].min()) / (data['Tahun'].max() - data['Tahun'].min())
data['Luas_Panen_Log'] = np.log1p(data['Luas Panen'])
data['Curah_Hujan_Log'] = np.log1p(data['Curah hujan'])
provinsi_stratify = data['Provinsi'].copy()

# One-hot encoding for Provinsi
data = pd.get_dummies(data, columns=['Provinsi'], drop_first=True)

# Additional interaction features
data['Hujan_Kelembapan'] = data['Curah hujan'] * data['Kelembapan']
data['Luas_Hujan_Ratio'] = data['Luas Panen'] / (data['Curah hujan'] + 1)
data['Suhu_Squared'] = data['Suhu rata-rata'] ** 2
data['Luas_Suhu_Interaction'] = data['Luas Panen'] * data['Suhu rata-rata']

# Prepare features and target
provinsi_columns = [col for col in data.columns if col.startswith('Provinsi_')]
feature_columns = ['Luas Panen', 'Curah hujan', 'Kelembapan', 'Suhu rata-rata', 'Hujan_Kelembapan', 
                   'Luas_Hujan_Ratio', 'Suhu_Squared', 'Luas_Suhu_Interaction', 
                   'Rata_Produksi_Provinsi', 'Tahun_Norm', 'Luas_Panen_Log', 'Curah_Hujan_Log'] + provinsi_columns
X = data[feature_columns]
y = np.log1p(data['Produksi'])

# Preprocessor for regression
reg_preprocessor = ColumnTransformer([
    ('num', RobustScaler(), ['Luas Panen', 'Curah hujan', 'Kelembapan', 'Suhu rata-rata', 'Hujan_Kelembapan', 
                            'Luas_Hujan_Ratio', 'Suhu_Squared', 'Luas_Suhu_Interaction', 
                            'Rata_Produksi_Provinsi', 'Tahun_Norm', 'Luas_Panen_Log', 'Curah_Hujan_Log']),
    ('cat', 'passthrough', provinsi_columns)
])

# GradientBoostingRegressor model
gbr_model = Pipeline([
    ('preprocessor', reg_preprocessor),
    ('regressor', GradientBoostingRegressor(random_state=42))
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=provinsi_stratify)

# Hyperparameter optimization for GradientBoostingRegressor
param_grid = {
    'regressor__n_estimators': [200, 300, 400],
    'regressor__learning_rate': [0.01, 0.05, 0.1],
    'regressor__max_depth': [3, 5, 7]
}
grid_search = GridSearchCV(gbr_model, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search.fit(X_train, y_train)
gbr_model = grid_search.best_estimator_
print(f"Parameter terbaik GradientBoosting: {grid_search.best_params_}")

# LinearRegression model
lr_model = Pipeline([
    ('preprocessor', reg_preprocessor),
    ('regressor', LinearRegression())
])
lr_model.fit(X_train, y_train)

# Evaluate models and store metrics
metrics = {}
for name, model in [('GradientBoosting', gbr_model), ('LinearRegression', lr_model)]:
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_test_orig = np.expm1(y_test)
    r2 = r2_score(y_test_orig, y_pred)

    # Weighted SMAPE
    def weighted_smape(y_true, y_pred):
        weights = np.where(y_true < 500000, 0.7, 0.3)
        smape = 100 * np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)) * weights)
        return smape
    smape = weighted_smape(y_test_orig, y_pred)
    accuracy_percentage = 100 - smape

    print(f"{name} - R²: {r2:.2f}, Accuracy: {accuracy_percentage:.2f}%")
    metrics[name] = {'r2': r2, 'accuracy_percentage': accuracy_percentage}

# Save residual data for visualization
y_pred_log = gbr_model.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_test_orig = np.expm1(y_test)
residuals = y_test_orig - y_pred
residual_data = [{'x': float(actual), 'y': float(residual)} for actual, residual in zip(y_test_orig, residuals)]
try:
    with open('static/residual_data.json', 'w') as f:
        json.dump(residual_data, f)
except Exception as e:
    print(f"Error menulis residual_data.json: {e}")

# Clustering
cluster_preprocessor = ColumnTransformer([
    ('num', RobustScaler(), ['Luas Panen', 'Curah hujan', 'Kelembapan', 'Suhu rata-rata', 'Hujan_Kelembapan', 
                            'Luas_Hujan_Ratio', 'Suhu_Squared', 'Luas_Suhu_Interaction', 
                            'Rata_Produksi_Provinsi', 'Tahun_Norm', 'Luas_Panen_Log', 'Curah_Hujan_Log']),
    ('cat', 'passthrough', provinsi_columns)
])
X_scaled = cluster_preprocessor.fit_transform(X)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
data['Cluster'] = kmeans.fit_predict(X_scaled)

# Define categories
cluster_means = data.groupby('Cluster')['Produksi'].mean().sort_values()
cluster_labels = {int(idx): label for idx, label in zip(cluster_means.index, ['Rendah', 'Sedang', 'Tinggi'])}  # Convert int32 to int
data['Kategori'] = data['Cluster'].map(cluster_labels)

# Save necessary data
provinsi_prod_map = data.groupby('Kategori')['Produksi'].mean().to_dict()
with open('provinsi_prod_map.json', 'w') as f:
    json.dump(provinsi_prod_map, f)

# Save feature columns and other metadata
metadata = {
    'feature_columns': feature_columns,
    'provinsi_columns': provinsi_columns,
    'tahun_min': int(data['Tahun'].min()),
    'tahun_max': int(data['Tahun'].max()),
    'provinsi_list': ['Aceh', 'Sumatera Utara', 'Sumatera Barat', 'Riau', 'Jambi', 'Sumatera Selatan', 'Bengkulu', 'Lampung'],
    'cluster_labels': cluster_labels,
    'metrics': metrics  # Save metrics for both models
}
with open('metadata.json', 'w') as f:
    json.dump(metadata, f)

# Save models and preprocessors
joblib.dump(gbr_model, 'gbr_model.pkl')
joblib.dump(lr_model, 'lr_model.pkl')
joblib.dump(cluster_preprocessor, 'cluster_preprocessor.pkl')
joblib.dump(kmeans, 'kmeans.pkl')
joblib.dump(data['Rata_Produksi_Provinsi'], 'rata_produksi_provinsi.pkl')

print("Training selesai dan model disimpan.")