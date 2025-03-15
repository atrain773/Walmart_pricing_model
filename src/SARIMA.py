import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Load the data
grouped_prices_df = pd.read_csv("../data/grouped_prices.csv")

# Convert date column to datetime
grouped_prices_df['date.x'] = pd.to_datetime(grouped_prices_df['date.x'])

# Aggregate data by CAT_ID and date
cat_sales = grouped_prices_df.groupby(['date.x', 'cat_id'])['sales'].sum().reset_index()

# Normalize CAT_IDs
cat_sales['cat_id'] = cat_sales['cat_id'].str.strip().str.upper()
categories = ["FOODS", "HOUSEHOLD", "HOBBIES"]

# Suggested SARIMA order
p, d, q = 1, 1, 1
P, D, Q, s = 1, 1, 1, 7  # Weekly seasonality

# Create subplots
fig, axes = plt.subplots(len(categories), 1, figsize=(12, 8), sharex=True)
fig.suptitle("SARIMA Forecasts for Each CAT_ID", fontsize=16)

# Forecast and plot for each CAT_ID
forecasts = {}
for i, cat_id in enumerate(categories):
    print(f"Processing CAT_ID: {cat_id}")

    # Filter data for the specific CAT_ID
    cat_data = cat_sales[cat_sales['cat_id'] == cat_id]
    if cat_data.empty:
        print(f"No data available for CAT_ID: {cat_id}. Skipping...")
        continue

    cat_data = cat_data.set_index('date.x')

    # Resample data to daily frequency
    daily_sales = cat_data['sales'].resample('D').sum().fillna(0)
    if daily_sales.sum() == 0:
        print(f"Insufficient sales data for CAT_ID: {cat_id}. Skipping...")
        continue

    try:
        # Fit the SARIMA model
        model = sm.tsa.statespace.SARIMAX(daily_sales,
                                          order=(p, d, q),
                                          seasonal_order=(P, D, Q, s),
                                          enforce_stationarity=False,
                                          enforce_invertibility=False)
        results = model.fit(disp=False)

        # Forecast the next 28 days
        forecast = results.get_forecast(steps=28)
        forecast_index = pd.date_range(start=daily_sales.index[-1] + pd.Timedelta(days=1), periods=28)
        forecasts[cat_id] = pd.Series(forecast.predicted_mean, index=forecast_index)

        # Plot historical data and forecast in subplot
        axes[i].plot(daily_sales, label='Historical Sales', color='blue')
        axes[i].plot(forecast_index, forecasts[cat_id], label='Forecast', color='red')
        axes[i].set_title(f"CAT_ID: {cat_id}")
        axes[i].set_ylabel("Sales")
        axes[i].grid()
        if i == len(categories) - 1:
            axes[i].set_xlabel("Date")
        axes[i].legend()

    except Exception as e:
        print(f"Failed to fit SARIMA for CAT_ID {cat_id}: {e}")
        continue

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# Combine all forecasts into a DataFrame
forecast_df = pd.DataFrame(forecasts)
forecast_df.index.name = 'date'
print(forecast_df.head())

# Save forecast results to CSV
forecast_df.to_csv('../data/cat_forecasts.csv')
