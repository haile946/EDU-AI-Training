import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import ssl
import urllib.request

# Disable SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

url = "https://raw.githubusercontent.com/haile946/EDU-AI-Training/main/Day2/fitness_data.csv"

try:
    df = pd.read_csv(url)
except Exception as e:
    print(f"Error loading from URL: {e}")
    # Create sample data if URL fails
    print("Creating sample data locally...")
    data = {
        'age': [22, 25, 30, 28, 35, 20, 27, 32, 24, 29, 31, 26, 33, 23, 34],
        'pushups_per_min': [50, 45, 35, 40, 30, 55, 42, 33, 48, 38, 37, 44, 32, 52, 36],
        'run_5km_minutes': [22, 23, 26, 24, 28, 21, 23, 27, 23, 25, 26, 22, 29, 20, 27],
        'fitness_level': ['excellent', 'good', 'average', 'good', 'average',
                          'excellent', 'good', 'average', 'excellent', 'good',
                          'average', 'good', 'average', 'excellent', 'average']
    }
    df = pd.DataFrame(data)

X = df[["age", "pushups_per_min", "run_5km_minutes"]]
y = df["fitness_level"]

model = DecisionTreeClassifier().fit(X, y)

print(
    "Prediction for a 24-year-old, 48 pushups, 23-min 5km:",
    model.predict([[24, 48, 23]])[0],
)
