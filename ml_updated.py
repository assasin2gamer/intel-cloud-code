import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from scipy.fft import fft
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

# Define channels, data, and labels
channels = list(range(1, 15))
data = np.random.randn(1000, 120, 14)
labels = np.random.randint(0, 2, 1000)

def interpolate():
    """
    Interpolate missing values in the data. (If data contains NaNs)
    """
    for i in channels:
        knn_model = KNeighborsRegressor(n_neighbors=3)
        knn_model.fit(data[i], data[i])
        data[i] = knn_model.predict(data[i])

def handle_outliers():
    """
    Handle outliers based on Interquartile Range (IQR).
    """
    for i in channels:
        curr = data[:, :, i]
        iqr = np.percentile(curr, 75) - np.percentile(curr, 25)
        fstQ = np.percentile(curr, 25)
        thdQ = np.percentile(curr, 75)
        
        # Replace outliers with the median of the data for that channel
        median_val = np.median(curr)
        for idx, x in enumerate(curr):
            if x > (thdQ + 1.5 * iqr) or x < (fstQ - 1.5 * iqr):
                data[idx, :, i] = median_val

def extract_features_from_channel(data):
    """
    Extract FFT features from EEG data.
    """
    fft_features = np.abs(fft(data, axis=1))
    delta_band = np.mean(fft_features[:, 0:4], axis=1).reshape(-1, 1)
    theta_band = np.mean(fft_features[:, 4:8], axis=1).reshape(-1, 1)
    alpha_band = np.mean(fft_features[:, 8:12], axis=1).reshape(-1, 1)
    beta_band = np.mean(fft_features[:, 12:30], axis=1).reshape(-1, 1)
    gamma_band = np.mean(fft_features[:, 30:45], axis=1).reshape(-1, 1)

    return np.hstack([delta_band, theta_band, alpha_band, beta_band, gamma_band])

# Extract features for each channel and combine
features_list = [extract_features_from_channel(data[:, :, i]) for i in channels]
features = np.hstack(features_list)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

# Define hyperparameter distributions
param_dist = {
    'n_estimators': list(range(50,500)),
    'max_depth': list(range(1,20))
}

# Randomized search for best hyperparameters
rf = RandomForestClassifier()
rand_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=5, cv=5)
rand_search.fit(X_train, y_train)
best_rf = rand_search.best_estimator_
