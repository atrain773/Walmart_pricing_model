import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

prices_df = pd.read_csv("../data/prices.csv")
grouped_data_df = pd.read_csv("../data/grouped_data.csv")

df = pd.merge(grouped_data_df, prices_df, on=['wm_yr_wk','cat_id'], how='left')

df['is_christmas'] = df['event_name_1.y'].apply(lambda x: 1 if x == 'Christmas' else 0)
df['is_thanksgiving'] = df['event_name_1.y'].apply(lambda x: 1 if x == 'Thanksgiving' else 0)

days_of_week = pd.get_dummies(df['weekday.x'], drop_first=True)
df = pd.concat([df, days_of_week], axis=1)

df['sales'] = df['total'] * df['price_wmean']

df.to_csv('../data/grouped_prices.csv', index=False)

# Ensure 'date.x' is in datetime format and sort by date
df['date.x'] = pd.to_datetime(df['date.x'])
df = df.sort_values(by='date.x')

# Get unique categories
categories = df['cat_id'].unique()

# Dictionary to store models for each category
models = {}

# Create subplots for actual vs predicted and residuals
plt.figure(figsize=(12, len(categories) * 8))  # Adjust figure size

for i, category in enumerate(categories):
    # Filter data for the current category
    category_data = df[df['cat_id'] == category]
    
    # Select features and target
    X = category_data[['price_wmean', 'is_christmas', 'is_thanksgiving'] + list(days_of_week.columns)]
    y = category_data['sales']
    
    # Add constant to the model
    X = sm.add_constant(X)
    
    # Train the model
    model = sm.OLS(y, X).fit()
    models[category] = model
    
    # Generate predictions
    category_data['predictions'] = model.predict(X)
    
    # Calculate residuals
    category_data['residuals'] = category_data['sales'] - category_data['predictions']

    # Calculate RMSE
    rmse = np.sqrt(np.mean((category_data['sales'] - category_data['predictions']) ** 2))
    print(f"RMSE for {category}: {rmse:.2f}")
    
    # Plot Actual vs Predicted Sales
    plt.subplot(len(categories), 2, i * 2 + 1)
    plt.plot(category_data['date.x'], category_data['sales'], label=f'Actual Sales - {category}', color='blue')
    plt.plot(category_data['date.x'], category_data['predictions'], label=f'Predicted Sales - {category}', color='orange', linestyle = '--')
    plt.title(f'Actual vs Predicted Sales for {category} \nRMSE: {rmse:.2f}')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    
    # Plot Residuals vs Original Sales
    plt.subplot(len(categories), 2, i * 2 + 2)
    plt.scatter(category_data['sales'], category_data['residuals'], color='red', alpha=0.6, s=10)
    plt.axhline(0, color='black', linewidth=1)
    plt.title(f'Residuals vs Original Sales for {category}')
    plt.xlabel('Original Sales')
    plt.ylabel('Residuals')

plt.tight_layout()
plt.show()

print(df.head())


