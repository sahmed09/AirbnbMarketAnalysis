import pandas as pd
import numpy as np

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend: saves figures instead of showing GUI windows
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

# Try to import CatBoost; if not installed, we'll just skip it silently
try:
    from catboost import CatBoostRegressor

    catboost_available = True
except ImportError:
    catboost_available = False

# ==========================
# 1. LOAD YOUR CLEANED DATA
# ==========================

df = pd.read_csv("listings_cleaned_no_garbage.csv")

print("Shape of cleaned data:", df.shape)
print(df.head())

# ==========================
# 2. BASIC CLEANING FOR PRICE ANALYSIS
# ==========================

df = df.copy()
df = df[df['price'].notna()]
df = df[df['price'] > 0]

# Remove extreme outliers (top 1% of price)
price_upper = df['price'].quantile(0.99)
df = df[df['price'] <= price_upper]

print("Shape after basic price filtering:", df.shape)

# ==========================
# 2.1 PRICE DISTRIBUTION PLOTS
# ==========================

plt.figure(figsize=(10, 6))
sns.histplot(df['price'], bins=50, kde=True)
plt.title("Price Distribution")
plt.xlabel("Price")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("01_price_distribution.png")
plt.close()

df['price_log'] = np.log(df['price'] + 1)

plt.figure(figsize=(10, 6))
sns.histplot(df['price_log'], bins=50, kde=True)
plt.title("Log(Price + 1) Distribution")
plt.xlabel("log(price + 1)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("02_price_log_distribution.png")
plt.close()

# PRICE vs ROOM TYPE
if 'room_type' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='room_type', y='price', data=df)
    plt.title("Price by Room Type")
    plt.xlabel("Room Type")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.savefig("03_price_by_room_type.png")
    plt.close()

# PRICE vs NEIGHBOURHOOD GROUP
if 'neighbourhood_group' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='neighbourhood_group', y='price', data=df)
    plt.title("Price by Neighbourhood Group")
    plt.xlabel("Neighbourhood Group")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.savefig("04_price_by_neighbourhood_group.png")
    plt.close()

# PRICE vs NUMBER OF REVIEWS
if 'number_of_reviews' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='number_of_reviews', y='price', data=df, alpha=0.5)
    plt.title("Price vs Number of Reviews")
    plt.xlabel("Number of Reviews")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.savefig("05_price_vs_number_of_reviews.png")
    plt.close()

# ==========================
# 2.2 CORRELATION HEATMAP (PRICE & FEATURES)
# ==========================

corr_df = df.copy()

# Encode categoricals as codes so we can include them in corr
if 'neighbourhood_group' in corr_df.columns:
    corr_df['neighbourhood_group_code'] = corr_df['neighbourhood_group'].astype('category').cat.codes
if 'room_type' in corr_df.columns:
    corr_df['room_type_code'] = corr_df['room_type'].astype('category').cat.codes
if 'neighbourhood' in corr_df.columns:
    corr_df['neighbourhood_code'] = corr_df['neighbourhood'].astype('category').cat.codes

corr_features = ['price', 'minimum_nights', 'number_of_reviews', 'availability_365']
for c in ['neighbourhood_group_code', 'room_type_code', 'neighbourhood_code']:
    if c in corr_df.columns:
        corr_features.append(c)

corr_matrix = corr_df[corr_features].corr(method='pearson')

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar_kws={"shrink": .8})
plt.title("Correlation Heatmap: Price & Features")
plt.tight_layout()
plt.savefig("10_correlation_heatmap_price_features.png")
plt.close()

# ==========================
# 3. PRICE PREDICTION MODEL
# ==========================

feature_candidates = [
    'minimum_nights',
    'number_of_reviews',
    'availability_365',
    'neighbourhood_group',
    'neighbourhood',
    'room_type'
]

features = [col for col in feature_candidates if col in df.columns]
print("Using features:", features)

model_df = df[features + ['price']].copy()

cat_cols = [c for c in ['neighbourhood_group', 'neighbourhood', 'room_type'] if c in model_df.columns]
num_cols = [c for c in model_df.columns if c not in cat_cols + ['price']]

# One-hot encode categoricals
model_df = pd.get_dummies(model_df, columns=cat_cols, drop_first=True)

X = model_df.drop(columns=['price'])
y = model_df['price']

print("X shape (features):", X.shape)
print("y shape (target):", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================
# 3.1 TRAIN MULTIPLE MODELS + METRICS
# ==========================

results = {}

# Linear models
linear_models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=0.5),
    "Lasso": Lasso(alpha=0.001),
    "ElasticNet": ElasticNet(alpha=0.001, l1_ratio=0.5)
}

for name, model in linear_models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2}
    print(f"--- {name} ---")
    print(f"MAE : {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R2  : {r2:.3f}")
    print()

# Tree-based models (for metrics AND feature importance)
etr = ExtraTreesRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
etr.fit(X_train_scaled, y_train)
y_pred_etr = etr.predict(X_test_scaled)

results["ExtraTrees"] = {
    "MAE": mean_absolute_error(y_test, y_pred_etr),
    "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_etr)),
    "R2": r2_score(y_test, y_pred_etr)
}

rf = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)

results["RandomForest"] = {
    "MAE": mean_absolute_error(y_test, y_pred_rf),
    "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_rf)),
    "R2": r2_score(y_test, y_pred_rf)
}

# CatBoost model (only if installed)
if catboost_available:
    cat_model = CatBoostRegressor(
        depth=6,
        learning_rate=0.05,
        n_estimators=300,
        loss_function='RMSE',
        verbose=False,
        random_seed=42
    )
    # Using scaled numeric features (already one-hot encoded)
    cat_model.fit(X_train_scaled, y_train)
    y_pred_cat = cat_model.predict(X_test_scaled)

    results["CatBoost"] = {
        "MAE": mean_absolute_error(y_test, y_pred_cat),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_cat)),
        "R2": r2_score(y_test, y_pred_cat)
    }

# Print added models' scores
for name in ["ExtraTrees", "RandomForest", "CatBoost"]:
    if name in results:
        print(f"--- {name} ---")
        print(f"MAE : {results[name]['MAE']:.3f}")
        print(f"RMSE: {results[name]['RMSE']:.3f}")
        print(f"R2  : {results[name]['R2']:.3f}")
        print()

# ==========================
# 3.2 MODEL COMPARISON TABLE (EXCEL)
# ==========================

metrics_df = pd.DataFrame(results).T  # models as rows
metrics_df = metrics_df[['MAE', 'RMSE', 'R2']]  # order columns

print("\nModel comparison:\n", metrics_df)

metrics_df.to_excel("model_comparison_metrics.xlsx", index=True)
print("\nSaved model comparison metrics to 'model_comparison_metrics.xlsx'.")

# ==========================
# 3.3 PLOT TRUE VS PREDICTED (BEST MODEL BY R²)
# ==========================

best_model_name = max(results.keys(), key=lambda m: results[m]['R2'])
print(f"\nBest model by R²: {best_model_name}")

# Get the fitted instance
if best_model_name in linear_models:
    best_model = linear_models[best_model_name]
elif best_model_name == "ExtraTrees":
    best_model = etr
elif best_model_name == "RandomForest":
    best_model = rf
elif best_model_name == "CatBoost" and catboost_available:
    best_model = cat_model
else:
    best_model = linear_models["LinearRegression"]  # fallback

y_pred_best = best_model.predict(X_test_scaled)

plt.figure(figsize=(7, 7))
plt.scatter(y_test, y_pred_best, alpha=0.4)
plt.xlabel("True Price")
plt.ylabel("Predicted Price")
plt.title(f"True vs Predicted Price ({best_model_name})")
max_val = max(y_test.max(), y_pred_best.max())
plt.plot([0, max_val], [0, max_val], 'r--')
plt.tight_layout()
plt.savefig("06_true_vs_predicted_best_model.png")
plt.close()

residuals = y_test - y_pred_best
plt.figure(figsize=(7, 5))
sns.histplot(residuals, bins=50, kde=True)
plt.title(f"Residuals Distribution ({best_model_name})")
plt.xlabel("Residual (True - Predicted)")
plt.tight_layout()
plt.savefig("07_residuals_best_model.png")
plt.close()

# ==========================
# 3.4 FEATURE IMPORTANCE (EXTRATREES, RANDOMFOREST, CATBOOST)
# ==========================

# ExtraTrees importance
importances_etr = etr.feature_importances_
feat_imp_etr = pd.Series(importances_etr, index=X.columns).sort_values(ascending=False)

top_n = 20
plt.figure(figsize=(10, 8))
feat_imp_etr.head(top_n).iloc[::-1].plot(kind='barh')
plt.title("Feature Importance (ExtraTreesRegressor)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("11_feature_importance_extratrees.png")
plt.close()

print("\nTop ExtraTrees features:")
print(feat_imp_etr.head(20))

# RandomForest importance
importances_rf = rf.feature_importances_
feat_imp_rf = pd.Series(importances_rf, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 8))
feat_imp_rf.head(top_n).iloc[::-1].plot(kind='barh')
plt.title("Feature Importance (RandomForestRegressor)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("12_feature_importance_randomforest.png")
plt.close()

print("\nTop RandomForest features:")
print(feat_imp_rf.head(20))

# CatBoost importance (if available)
if catboost_available:
    importances_cat = cat_model.feature_importances_
    feat_imp_cat = pd.Series(importances_cat, index=X.columns).sort_values(ascending=False)

    plt.figure(figsize=(10, 8))
    feat_imp_cat.head(top_n).iloc[::-1].plot(kind='barh')
    plt.title("Feature Importance (CatBoostRegressor)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("13_feature_importance_catboost.png")
    plt.close()

    print("\nTop CatBoost features:")
    print(feat_imp_cat.head(20))

# ==========================
# 4. NEIGHBOURHOOD CLUSTERING
# ==========================

if 'neighbourhood' in df.columns:
    group_cols = ['neighbourhood']
    agg_dict = {'price': 'mean'}

    if 'number_of_reviews' in df.columns:
        agg_dict['number_of_reviews'] = 'mean'
    if 'minimum_nights' in df.columns:
        agg_dict['minimum_nights'] = 'mean'
    if 'availability_365' in df.columns:
        agg_dict['availability_365'] = 'mean'

    neigh_stats = df.groupby(group_cols).agg(agg_dict).reset_index()
    neigh_stats = neigh_stats.dropna()

    print("Neighbourhood stats shape:", neigh_stats.shape)
    print(neigh_stats.head())

    cluster_features = ['price']
    for extra_col in ['number_of_reviews', 'minimum_nights', 'availability_365']:
        if extra_col in neigh_stats.columns:
            cluster_features.append(extra_col)

    X_neigh = neigh_stats[cluster_features].values

    scaler_neigh = StandardScaler()
    X_neigh_scaled = scaler_neigh.fit_transform(X_neigh)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    neigh_stats['cluster'] = kmeans.fit_predict(X_neigh_scaled)

    print(neigh_stats.head())

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='cluster', y='price', data=neigh_stats)
    plt.title("Neighbourhood Clusters by Average Price")
    plt.xlabel("Cluster")
    plt.ylabel("Average Price")
    plt.tight_layout()
    plt.savefig("08_neighbourhood_clusters_price_boxplot.png")
    plt.close()

    top_n = 30
    top_neigh = neigh_stats.sort_values('price', ascending=False).head(top_n)

    plt.figure(figsize=(12, 8))
    sns.barplot(
        x='price',
        y='neighbourhood',
        hue='cluster',
        data=top_neigh,
        dodge=False
    )
    plt.title(f"Top {top_n} Neighbourhoods by Average Price (Clustered)")
    plt.xlabel("Average Price")
    plt.ylabel("Neighbourhood")
    plt.tight_layout()
    plt.savefig("09_top_neighbourhoods_by_price_clusters.png")
    plt.close()

else:
    print("Column 'neighbourhood' not found, skipping neighbourhood clustering.")

#
# # ==========================
# # XGBOOST REGRESSION MODEL
# # ==========================
#
# from xgboost import XGBRegressor
#
# xgb = XGBRegressor(
#     n_estimators=400,
#     max_depth=6,
#     learning_rate=0.05,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     objective="reg:squarederror",
#     random_state=42,
#     n_jobs=-1
# )
#
# # Fit model
# xgb.fit(X_train_scaled, y_train)
#
# # Predict
# y_pred_xgb = xgb.predict(X_test_scaled)
#
# # Metrics
# xgb_mae = mean_absolute_error(y_test, y_pred_xgb)
# xgb_rmse = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
# xgb_r2 = r2_score(y_test, y_pred_xgb)
#
# print("\n--- XGBoost Regression ---")
# print(f"MAE : {xgb_mae:.3f}")
# print(f"RMSE: {xgb_rmse:.3f}")
# print(f"R2  : {xgb_r2:.3f}")
#
# # Save to results dictionary (if using model comparison table)
# results["XGBoost"] = {
#     "MAE": xgb_mae,
#     "RMSE": xgb_rmse,
#     "R2": xgb_r2
# }
#
# # ==========================
# # XGBOOST FEATURE IMPORTANCE
# # ==========================
#
# importances_xgb = xgb.feature_importances_
# feat_imp_xgb = pd.Series(importances_xgb, index=X.columns).sort_values(ascending=False)
#
# plt.figure(figsize=(10, 8))
# feat_imp_xgb.head(20).iloc[::-1].plot(kind='barh')
# plt.title("Feature Importance (XGBoost)")
# plt.xlabel("Importance")
# plt.tight_layout()
# plt.savefig("13_feature_importance_xgboost.png")
# plt.close()
#
# print("\nTop XGBoost Features:")
# print(feat_imp_xgb.head(20))
