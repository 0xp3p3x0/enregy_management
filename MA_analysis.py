import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_and_prepare_data(filepath):
    data = pd.read_csv(filepath)
    data['DateTime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], errors='coerce')
    data = data.drop(columns=['Unnamed: 0', 'Date', 'Time'])
    data = data.dropna(subset=['DateTime', 'avg']).reset_index(drop=True)
    data['min'] = pd.to_numeric(data['min'], errors='coerce')
    data['avg'] = pd.to_numeric(data['avg'], errors='coerce')
    data['max'] = pd.to_numeric(data['max'], errors='coerce')
    return data

def descriptive_stats(data):
    hourly_data = data.resample('D', on='DateTime').mean()
    stats = {
        "Mean Consumption": hourly_data['avg'].mean(),
        "Median Consumption": hourly_data['avg'].median(),
        "Standard Deviation": hourly_data['avg'].std(),
        "Variance": hourly_data['avg'].var(),
        "Peak Demand Date": hourly_data['avg'].idxmax(),
        "Peak Avg Consumption": hourly_data['avg'].max(),
    }
    return stats, hourly_data

def train_ai_model(data):
    data['hour'] = data['DateTime'].dt.hour
    X = data[['hour', 'min', 'max']]
    y = data['avg']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor()
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "R²": r2_score(y_test, y_pred)
        }
   
    best_model_name = max(results, key=lambda k: results[k]['R²'])
    best_model = models[best_model_name]
    
    return best_model, results

def visualize(data, daily_data, stats):
    plt.figure(figsize=(12, 6))
    plt.plot(daily_data.index, daily_data['avg'], label='Average Daily Consumption', marker='o')
    plt.axhline(stats["Mean Consumption"], color='r', linestyle='--', label='Mean Consumption')
    plt.axhline(stats["Median Consumption"], color='g', linestyle=':', label='Median Consumption')
    plt.scatter(stats["Peak Demand Date"], stats["Peak Avg Consumption"], color='orange', label='Peak Demand', s=100)
    plt.xlabel('Date')
    plt.ylabel('Average Energy Consumption')
    plt.title('Daily Energy Consumption Trends')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    sns.histplot(daily_data['avg'], kde=True, bins=30, color='blue', edgecolor='black')
    plt.axvline(stats["Mean Consumption"], color='r', linestyle='--', label='Mean Consumption')
    plt.axvline(stats["Median Consumption"], color='g', linestyle=':', label='Median Consumption')
    plt.xlabel('Average Daily Consumption')
    plt.ylabel('Frequency')
    plt.title('Distribution of Daily Energy Consumption')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Load and prepare data
    file_path = 'CSV_MA.csv'
    data = load_and_prepare_data(file_path)
    
    # Perform descriptive statistics
    stats, daily_data = descriptive_stats(data)
    print("Descriptive Statistics:", stats)
    
    # Train AI-driven optimization model
    best_model, model_results = train_ai_model(data)
    print("Model Results:", model_results)
    print("Best Model:", best_model)
    
    # Visualize results
    visualize(data, daily_data, stats)
