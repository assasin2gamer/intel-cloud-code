import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from scipy.fft import fft
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import psycopg2
import pickle
from sklearn.metrics import mean_squared_error
import json
import re

conn = psycopg2.connect('postgresql://mindsdb:k7zLDNTQDme3wI6KpquUTQ@good-stag-12544.7tt.cockroachlabs.cloud:26257/defaultdb?sslmode=verify-full')
with conn.cursor() as cur:
    print(cur.execute("USE defaultdb"))
    print(cur.execute("SELECT * FROM eeg_data"))
    rows = cur.fetchall()
    for row in rows:
        json_data = row[1]
        numbers = [float(num) for num in re.findall(r'\d+.\d+|\d+', json_data)]

        print(numbers)