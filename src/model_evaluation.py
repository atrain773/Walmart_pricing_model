import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, DateFormatter


prices_df = pd.read_csv("../data/prices.csv")
grouped_data_df = pd.read_csv("../data/grouped_data_test.csv")

df = pd.merge(grouped_data_df, prices_df, on=['wm_yr_wk','cat_id'], how='left')

df['is_christmas'] = df['event_name_1'].apply(lambda x: 1 if x == 'Christmas' else 0)
df['is_thanksgiving'] = df['event_name_1'].apply(lambda x: 1 if x == 'Thanksgiving' else 0)

days_of_week = pd.get_dummies(df['weekday'], drop_first=True)
df = pd.concat([df, days_of_week], axis=1)

df['sales'] = df['total'] * df['price_wmean']

df.to_csv('../data/grouped_prices_test.csv', index=False)


# Load the data
prices_df = pd.read_csv("../data/grouped_prices_test.csv")
cat_forecast_df = pd.read_csv("../data/cat_forecasts.csv")

# Convert date column to datetime
prices_df['date'] = pd.to_datetime(prices_df['date'])
prices_df = prices_df.sort_values(by='date')

# Get unique categories
categories = prices_df['cat_id'].unique()

# Ensure predictions align with categories and dates
for category in categories:
    if category in cat_forecast_df.columns:
        prices_df.loc[prices_df['cat_id'] == category, 'predictions'] = cat_forecast_df[category].values
    else:
        print(f"Forecast data missing for category: {category}")

# Initialize the plot
plt.figure(figsize=(12, len(categories) * 8))  # Adjust figure size

for i, category in enumerate(categories):
    category_data = prices_df[prices_df['cat_id'] == category]

    if 'predictions' not in category_data.columns or category_data['predictions'].isnull().all():
        print(f"No predictions available for category: {category}")
        continue

    # Calculate residuals
    category_data['residuals'] = category_data['sales'] - category_data['predictions']

    # Calculate RMSE
    rmse = np.sqrt(np.mean((category_data['sales'] - category_data['predictions']) ** 2))
    print(f"RMSE for {category}: {rmse:.2f}")

    # Plot Actual vs Predicted Sales
    ax1 = plt.subplot(len(categories), 2, i * 2 + 1)
    ax1.plot(category_data['date'], category_data['sales'], label=f'Actual Sales - {category}', color='blue')
    ax1.plot(category_data['date'], category_data['predictions'], label=f'Predicted Sales - {category}', color='orange', alpha=0.6)
    ax1.set_title(f'Actual vs Predicted Sales for {category}\nRMSE: {rmse:.2f}')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Sales')
    ax1.legend()
    ax1.grid()

    # Format x-axis for better readability
    ax1.xaxis.set_major_locator(AutoDateLocator())  # Automatically space dates
    ax1.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))  # Format date as YYYY-MM-DD
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels

    # Plot Residuals vs Original Sales
    ax2 = plt.subplot(len(categories), 2, i * 2 + 2)
    ax2.scatter(category_data['sales'], category_data['residuals'], color='red', alpha=0.6, s=10)
    ax2.axhline(0, color='black', linewidth=1)
    ax2.set_title(f'Residuals vs Original Sales for {category}')
    ax2.set_xlabel('Original Sales')
    ax2.set_ylabel('Residuals')

plt.tight_layout()
plt.show()
