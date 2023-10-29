import ray
from fastapi import Request, FastAPI
from ray import serve
import time
import pandas as pd
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from scipy.fft import fft
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV


app = FastAPI()
ray.init()
serve.start(detached=True)


async def parse_req(request: Request):
    data = await request.json()
    target = data.get('target', None)
    di = json.loads(data['df'])
    df = pd.DataFrame(di)
    return df, target


@serve.deployment(route_prefix="/my_model")
@serve.ingress(app)
class MyModel:
    @app.post("/train")
    async def train(self, request: Request):
        df, target = await parse_req(request)
        feature_cols = list(set(list(df.columns)) - set([target]))
        self.feature_cols = feature_cols
        X = df.loc[:, self.feature_cols]
        Y = list(df[target])


        def interpolate():
             #Interpolate missing values in the data. (If data contains NaNs)
            for i in feature_cols:
                knn_model = KNeighborsRegressor(n_neighbors=3)
                knn_model.fit(X[i], X[i])
                X[i] = knn_model.predict(X[i])

        def handle_outliers():
            """
            Handle outliers based on Interquartile Range (IQR).
            """
            for i in range(len(feature_cols)):
                curr = X[:, :, i]
                iqr = np.percentile(curr, 75) - np.percentile(curr, 25)
                fstQ = np.percentile(curr, 25)
                thdQ = np.percentile(curr, 75)
                # Replace outliers with the median of the data for that channel
                median_val = np.median(curr)
                for x in range(len(curr)):
                    for y in range(len(curr[x])):
                        if curr[x][y] > (thdQ + 1.5 * iqr) or curr[x][y] < (fstQ - 1.5 * iqr):
                            curr[x][y] = median_val
                X[:,:,i] = curr

        def extract_features_from_channel(data):
            #Extract FFT features from EEG data.        
            fft_features = np.abs(fft(data, axis=1))
            delta_band = np.mean(fft_features[:, 0:4], axis=1).reshape(-1, 1)
            theta_band = np.mean(fft_features[:, 4:8], axis=1).reshape(-1, 1)
            alpha_band = np.mean(fft_features[:, 8:12], axis=1).reshape(-1, 1)
            beta_band = np.mean(fft_features[:, 12:30], axis=1).reshape(-1, 1)
            gamma_band = np.mean(fft_features[:, 30:45], axis=1).reshape(-1, 1)

            return np.hstack([delta_band, theta_band, alpha_band, beta_band, gamma_band])

        interpolate()
        handle_outliers()
        # Extract features for each channel and combine
        features_list = [extract_features_from_channel(X[:, :, i]) for i in range(len(feature_cols))]
        features = np.hstack(features_list)

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2)

        # Define hyperparameter distributions
        param_dist = {
            'n_estimators': list(range(50,500)),
            'max_depth': list(range(1,20))
        }

        # Randomized search for best hyperparameters
        rf = RandomForestClassifier()
        rand_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=5, cv=5)
        rand_search.fit(X_train, y_train)
        self.model = rand_search.best_estimator_

        return {'status': 'ok'}

    @app.post("/predict")
    async def predict(self, request: Request):
        df, _ = await parse_req(request)
        X = df.loc[:, self.feature_cols]
        predictions = self.model.predict(X)
        pred_dict = {'prediction': [float(x) for x in predictions]}
        return pred_dict


MyModel.deploy()

while True:
    time.sleep(1)