## Import Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import time
from sklearn.cross_decomposition import PLSRegression
from matplotlib.pyplot import subplots
from sklearn.linear_model import ElasticNet
from numpy import arange
from sklearn.linear_model import ElasticNetCV
import seaborn as sns
import sys
import os
log_path = os.path.expanduser('~/sta5703/elasticnet_log.txt')
os.makedirs(os.path.dirname(log_path), exist_ok=True)  
sys.stdout = open(log_path, 'w')


df = pd.read_csv('~/sta5703/maize.csv')
df.rename(columns=lambda x: x.strip(), inplace=True)

df.head()

df.shape

# Check null value in row
rows_with_nulls = df.isnull().any(axis=1).sum()
print("Number of rows with null values:", rows_with_nulls)

# Drop rows with any null values
data_cleaned = df.dropna()

data_cleaned.shape

# Identify categorical columns
categorical_columns = data_cleaned.select_dtypes(include=['object']).columns
print(categorical_columns)

from sklearn.preprocessing import LabelEncoder
# Create and fit the LabelEncoder
le = LabelEncoder()
data_cleaned['Geno_Code'] = le.fit_transform(data_cleaned['Geno_Code'].astype(str))

# Print the encoded DataFrame to see the result
print(data_cleaned.head())

data_cleaned.info()

data_cleaned.describe()

dtoa_stats = data_cleaned['DtoA'].describe()
print(dtoa_stats)

# Create a histogram with KDE
sns.histplot(data_cleaned['DtoA'], kde=True, bins=50, color='green', edgecolor='black')

# Adding title and labels
plt.title('Data Distribution with KDE')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.savefig("data_dist.png")
plt.show()

X = data_cleaned.drop(['DtoA'], axis = 1)
Y = data_cleaned['DtoA']
X.head()

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

## Elastic_Net

# Define cross-validation strategy
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Define the ranges for alphas and l1_ratio
alphas = np.logspace(-3, 3, 50)
l1_ratios = np.arange(0.0, 1.0, 0.02)

# Initialize and fit the Scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Starting Parameter Tuning (ElasticNetCV) and Training")
start_time_train = time.time()

# ElasticNetCV initialization
elasticnet_model = ElasticNetCV(alphas=alphas, l1_ratio=l1_ratios, cv=cv, max_iter=10000, random_state=42, n_jobs=-1)

# Fit the model on the SCALED training data
elasticnet_model.fit(X_train_scaled, Y_train)

#Training Time Calculation
end_time_train = time.time()
training_time = end_time_train - start_time_train

# Extracting Results

# Extracting cofficients, feature names, counts
coef = elasticnet_model.coef_
feat = X_train.columns if hasattr(X_train, "columns") else [f"x{i}" for i in range(X_train.shape[1])]
n_features = X_train.shape[1]


# Create a DataFrame to combine feature details and coefficients
full_coef_df = pd.DataFrame({'Feature_Name': feat, 'Feature_Number': np.arange(1, n_features + 1), 'Coefficient': coef})

sorted_coef_df_full = full_coef_df.sort_values(by='Coefficient', key=abs, ascending=False)

print("\n--- All Features and Their Coefficients (Sorted by Magnitude) ---")
print(sorted_coef_df_full.to_string(index=False, float_format='%.4f'))

# Used Feature Calculation 

tolerance = 1e-4

active_features_df = full_coef_df[np.abs(full_coef_df['Coefficient']) > tolerance].copy()

# Count the number of active features from the filtered DataFrame
active_features_count = len(active_features_df)

# Count the total number of features (already defined as n_features)
total_features = n_features

# Print the feature usage summary
print("\n--- Model Feature Usage ---")
print(f"Total features considered: {total_features}")
print(f"Number of features used: {active_features_count}")
print(f"Number of features set to zero: {total_features - active_features_count}")

active_features_sorted = active_features_df.sort_values(by='Coefficient', key=abs, ascending=False).reset_index(drop=True)

print("\n--- Active Features and Their Non-Zero Coefficients")
print(active_features_sorted.to_string(index=False, float_format='%.4f'))

# Best alpha, best l1_ratio and MSE
best_alpha = elasticnet_model.alpha_
best_l1_ratio = elasticnet_model.l1_ratio_
best_val_mse = mean_squared_error(Y_train, elasticnet_model.predict(X_train_scaled))

# Plot Alpha vs MSE

# Get the index of the best l1_ratio found
best_l1_ratio_index = np.where(elasticnet_model.l1_ratio == best_l1_ratio)[0][0]

# Use the alphas from the fitted model for plotting
alphas_plot = elasticnet_model.alphas_

# Calculate mean and std of MSE *across CV folds* (axis=2) for *all* l1_ratios
full_mean_mse = elasticnet_model.mse_path_.mean(axis=2)
full_std_mse = elasticnet_model.mse_path_.std(axis=2)

# Select the mean and std corresponding to the best l1_ratio
mean_mse = full_mean_mse[best_l1_ratio_index, :]
std_mse = full_std_mse[best_l1_ratio_index, :]

plt.figure(figsize=(8, 5))
plt.errorbar(np.log10(alphas_plot), mean_mse, yerr=std_mse, fmt='o', color='red', ecolor='lightgray', elinewidth=2, capsize=2)
plt.axvline(np.log10(best_alpha), color='darkblue', linestyle='--', label=f"Best α={best_alpha:.4f}")
plt.xlabel("log(α)")
plt.ylabel("Mean-Squared Error")
plt.legend()
plt.savefig("mse_vs_alpha_plot.png")
plt.show()

# Plot l1_ratio vs MSE 

best_alpha_index = np.where(elasticnet_model.alphas_ == elasticnet_model.alpha_)[0][0]

# Extract the MSE path data: mse_path_ shape is (n_l1_ratio, n_alphas, n_folds)
# Select the data for the 'best_alpha_index' across all l1_ratios
# Resulting array shape is (n_l1_ratio, n_folds)

mse_for_best_alpha = elasticnet_model.mse_path_[:, best_alpha_index, :]

# Calculate the mean and standard deviation of MSE across the CV folds (axis=1)
mean_mse_vs_l1_ratio = mse_for_best_alpha.mean(axis=1)
std_mse_vs_l1_ratio = mse_for_best_alpha.std(axis=1)

# Define the l1_ratios for plotting (same as the index 0 of mse_path_)
l1_ratios_plot = elasticnet_model.l1_ratio # Uses the l1_ratios array passed to ElasticNetCV

plt.figure(figsize=(8, 5))
plt.errorbar(l1_ratios_plot, mean_mse_vs_l1_ratio, yerr=std_mse_vs_l1_ratio, fmt='o', color='green', ecolor='lightcoral', elinewidth=2, capsize=2)
plt.axvline(best_l1_ratio, color='darkred', linestyle='--', label=f"Best $l_1$ ratio={best_l1_ratio:.4f}")
plt.xlabel("$l_1$ ratio")
plt.ylabel("Mean CV-Squared Error")
plt.title(f"CV-MSE vs. $l_1$ ratio (with $\\alpha$ fixed at $\\alpha={best_alpha:.4g}$)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig("cv_mse_vs_l1_ratio_plot_from_path.png")
plt.show()

# Plot CV-MSE vs. combination of l1_ratio and alpha (Contour Plot)

# Average the MSE over the K folds (axis=2) to get the 2D MSE surface. The resulting shape is (n_l1_ratio, n_alphas)
mean_cv_mse_surface = elasticnet_model.mse_path_.mean(axis=2)

# Define the plotting grid (X and Y coordinates)
l1_ratios_plot = elasticnet_model.l1_ratio
log_alphas_plot = np.log10(elasticnet_model.alphas_)

# Create the plot
plt.figure(figsize=(10, 8))
contour = plt.contourf(log_alphas_plot, l1_ratios_plot, mean_cv_mse_surface, levels=20, cmap='viridis')
plt.colorbar(contour, label='Mean Cross-Validation MSE')
plt.plot(np.log10(elasticnet_model.alpha_), elasticnet_model.l1_ratio_, marker='*', color='red', markersize=15, label=f"Optimum: $\\alpha={elasticnet_model.alpha_:.4g}$, $l_1$ ratio={elasticnet_model.l1_ratio_:.4f}")
plt.xlabel("log($\\alpha$) (Regularization Strength)")
plt.ylabel("$l_1$ ratio (Mixing Parameter)")
plt.title("Contour Plot of CV-MSE vs. $\\alpha$ and $l_1$ ratio")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig("cv_mse_contour_plot.png")
plt.show()


# Coefficient Shrinkage Path
alphas_path = elasticnet_model.alphas_
coefs = []
for a in alphas_path:
    enet = ElasticNet(alpha=a, l1_ratio=best_l1_ratio, max_iter=10000, random_state=42)
    enet.fit(X_train_scaled, Y_train)
    coefs.append(enet.coef_)
coefs = np.array(coefs)

# Plotting the coefficients
plt.figure(figsize=(8, 5))
plt.plot(np.log10(alphas_path), coefs)
plt.axvline(np.log10(best_alpha), color='darkblue', linestyle='--', label="Best α")
plt.axhline(0, color='k', linestyle='--', lw=1)
plt.xlabel("log(α)")
plt.ylabel("Coefficients")
plt.savefig("coefficient_plot.png")
plt.show()

# Final Evaluation on Test Data
start_time_test = time.time()
y_test_pred = elasticnet_model.predict(X_test_scaled)

r_square = r2_score(Y_test, y_test_pred)
mse = mean_squared_error(Y_test, y_test_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(Y_test, y_test_pred)

end_time_test = time.time()
test_prediction_time = end_time_test - start_time_test

# Print all
print(f"Training Time: {training_time:.4f} seconds")
print(f"Best alpha: {best_alpha:.4g}")
print(f"Best l1_ratio: {best_l1_ratio:.4g}")
print(f"total features selected in elasticnet: {n_features}")
print(f"active features: {active_features_count}")
print(f"Validation MSE at best alpha: {best_val_mse:,.4f}")
print(f"Test Prediction Time: {test_prediction_time:.4f} seconds")
print(f'R2 Score = {r_square:.4f}')
print(f"Test MSE : {mse:,.4f}")
print(f"Test RMSE: {rmse:,.4f}")
print(f"Test MAE : {mae:,.4f}")

## PCR

# Start Time for Training
start_time_train = time.time()

# GridSearchCV
pcr_model = Pipeline([('scaler', StandardScaler()), ('pca', PCA()), ('lr', LinearRegression())])

# Define the range of k (number of components) to test
param_grid_pca = {'pca__n_components': np.arange(1, 51, 1)}
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Set up GridSearchCV (5-fold CV)
grid_search = GridSearchCV(estimator=pcr_model, param_grid=param_grid_pca, scoring='neg_mean_squared_error', cv=cv)

# Run the search
grid_search.fit(X_train, Y_train)

# Stop timer
end_time_train = time.time()
training_time_PCR = end_time_train - start_time_train

# Extract optimal k and best MSE
optimal_k = grid_search.best_params_['pca__n_components']
best_cv_mse = -grid_search.best_score_

print(f"Time taken for GridSearchCV: {training_time_PCR:.4f} seconds")
print(f"Optimal Number of Principal Components (k): {optimal_k}")
print(f"Best 5-Fold CV MSE: {best_cv_mse:.4f}")

## Plotting MSE vs. k
cv_mse_scores_pca = -grid_search.cv_results_['mean_test_score']
cv_std_pca = grid_search.cv_results_['std_test_score'] / np.sqrt(cv.get_n_splits())
k_range = param_grid_pca['pca__n_components']
plt.figure(figsize=(8, 5))
plt.errorbar(k_range, cv_mse_scores_pca, cv_std_pca, marker='o', linestyle='-')
plt.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal k = {optimal_k}')
plt.xlabel('Number of Principal Components (k)')
plt.ylabel('Mean Squared Error (MSE)')
# plt.title('PCA: MSE vs. Optimal K (GridSearchCV)')
plt.legend()
plt.show()

## PCA Scores and Top Features
best_pca_model = grid_search.best_estimator_.named_steps['pca']
scaler_model = grid_search.best_estimator_.named_steps['scaler']
explained_variance = best_pca_model.explained_variance_ratio_

print("\nPCA Score (Explained Variance) per Component")
for i, var in enumerate(explained_variance):
    cumulative_variance = np.sum(explained_variance[:i+1])
    print(f"PC {i+1}: {var:.4f} (Cumulative: {cumulative_variance:.4f})")

# Determine Top Original Features contributing to PC1
loadings_pc1 = pd.Series(best_pca_model.components_[0], index=X_train.columns)

print("\n Top 10 Original Features based on PC1 Loading")
top_features = loadings_pc1.abs().sort_values(ascending=False).head(10)
print(top_features)

# Final Model Testing and Metrics
final_pcr_model = grid_search.best_estimator_

# Start timer for prediction
start_time_test = time.time()

# Predict on the holdout test set
y_pred_test = final_pcr_model.predict(X_test)

# Evaluation Metrics
mse = mean_squared_error(Y_test, y_pred_test)
rmse = np.sqrt(mse)
r2 = r2_score(Y_test, y_pred_test)
mae = mean_absolute_error(Y_test, y_pred_test)

# Stop timer
end_time_test = time.time()
test_prediction_time_PCR = end_time_test - start_time_test

print("\n--- Final PCR Model Evaluation Metrics ---")
print(f"Time taken for Prediction: {test_prediction_time_PCR:.4f} seconds")
print(f"Features (Components) Used: {optimal_k}")
print(f"MSE (Mean Squared Error):       {mse:.4f}")
print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")
print(f"R^2 Score:                      {r2:.4f}")
print(f"MAE (Mean Absolute Error):      {mae:.4f}")

## PLS

pls_model =  PLSRegression(n_components=2, scale=True)

# Start Time for Training
start_time_train_pls = time.time()

# model training for CV
param_grid = {'n_components':range(1, 30)}
grid_pls = GridSearchCV(pls_model, param_grid, cv=cv, scoring='neg_mean_squared_error')
grid_pls.fit(X_train, Y_train)

# Extract optimal k and best MSE
optimal_n_component = grid_pls.best_params_['n_components']
optimal_mse_pls = -grid_pls.best_score_

print(f"Optimal Number of n Components : {optimal_n_component}")
print(f"Optimal MSE in Grid_search: {optimal_mse_pls:.4f}")

plt.figure(figsize=(8, 6))
n_comp = param_grid['n_components']
plt.errorbar(n_comp,-grid_pls.cv_results_['mean_test_score'], grid_pls.cv_results_['std_test_score'] / np.sqrt(cv.get_n_splits()), marker='o')
plt.axvline(x=optimal_n_component, color='r', linestyle='--', label=f'Optimal n_component = {optimal_n_component}')
plt.xlabel('Number of Components (k)')
plt.ylabel('Mean Squared Error (MSE)')
# plt.title('PCR: MSE vs. Number of Components (GridSearchCV)')
plt.legend()
plt.show()

# final model training
final_pls_model =  PLSRegression(n_components=1, scale=True)
final_pls_model.fit(X_train, Y_train)

# Stop timer
end_time_train_pls = time.time()
training_time_pls = end_time_train_pls - start_time_train_pls

# Feature names
features = X_train.columns.tolist() if hasattr(X_train, "columns") else [f"x{i}" for i in range(X_train.shape[1])]

# Coefficients and intercept
coef = final_pls_model.coef_.ravel()
intercept = float(np.ravel(final_pls_model.intercept_)[0])

# Table of features + coefficients, sorted by |coef|
coef_df = pd.DataFrame({"feature": features, "coef": np.round(coef,4), "abs_coef": np.round(np.abs(coef),4)}).sort_values("abs_coef", ascending=False).reset_index(drop=True)

print("Intercept:", intercept)
print(coef_df.head(10))

# Start timer for prediction
start_time_test_pls = time.time()

#prediction on test set
Y_pred_test_pls = final_pls_model.predict(X_test)

# Evaluation Metrics
mse_pls = mean_squared_error(Y_test, Y_pred_test_pls)
rmse_pls = np.sqrt(mse_pls)
r2_pls = r2_score(Y_test, Y_pred_test_pls)
mae_pls = mean_absolute_error(Y_test, Y_pred_test_pls)

# Stop timer
end_time_test_pls = time.time()
test_prediction_time_pls = end_time_test_pls - start_time_test_pls

# print all result:
print("\n--- Final PCR Model Evaluation Metrics ---")
print(f"Time taken for model training: {training_time_pls:.4f} seconds")
print(f"Time taken for Prediction: {test_prediction_time_pls:.4f} seconds")
print(f"MSE of pls(Mean Squared Error):       {mse_pls:.4f}")
print(f"RMSE of pls (Root Mean Squared Error): {rmse_pls:.4f}")
print(f"R^2 Score of pls :                      {r2_pls:.4f}")
print(f"MAE of pls (Mean Absolute Error):      {mae_pls:.4f}")

sys.stdout.close() 
