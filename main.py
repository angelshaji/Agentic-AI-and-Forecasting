import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA


# =====================================================
# STEP 1: LOAD THE DATA
# =====================================================

df = pd.read_csv("data/train.csv")

# =====================================================
# STEP 2: FILTER TO ONE STORE (ALL ITEMS)
# taking column "store" from the data and filtering for store 1 
# =====================================================

store_df = df[df["store"] == 1].copy()


# =====================================================
# STEP 3: PREPARE DATES
# turning the date column into readable dates for python and adding a month column which 
#will be used to turn daily sales data into monthly data 
# =====================================================

store_df["date"] = pd.to_datetime(store_df["date"])
store_df["month"] = store_df["date"].dt.to_period("M")


# =====================================================
# STEP 4: AGGREGATE DAILY → MONTHLY REVENUE
# =====================================================

monthly_df = (
    store_df
    .groupby("month")["sales"]
    .sum()
    .reset_index()
)

monthly_df.columns = ["Month", "Revenue"]
monthly_df["Month"] = monthly_df["Month"].dt.to_timestamp()

print("Monthly revenue (store 1, all items):")
print(monthly_df.head())


# =====================================================
# STEP 5: FILTER TO LAST 12 MONTHS
# I am keeping only the last 12 months of data for short-term quarterly forecasting
# =====================================================

# Keep only the most recent 12 months of data
last_12_df = monthly_df.tail(12)

print("\nLast 12 months of revenue data:")
print(last_12_df)


# =====================================================
# STEP 6: TRAIN ARIMA MODEL (SHORT-TERM FORECASTING)
# =====================================================

# For short-term (quarterly) data:
# - Seasonality is not reliable
# - Use simple ARIMA instead
model = ARIMA(
    last_12_df["Revenue"],
    order=(1, 1, 0)
)

model_fit = model.fit()


# =====================================================
# STEP 7: FORECAST NEXT QUARTER
# =====================================================

forecast_steps = 3
forecast = model_fit.forecast(steps=forecast_steps)

print("\nForecasted revenue for next quarter:")
print(forecast)

# =====================================================
# STEP 8: VISUALIZE QUARTERLY FORECAST
# =====================================================

last_month = last_12_df["Month"].iloc[-1]
future_months = pd.date_range(
    start=last_month + pd.offsets.MonthBegin(),
    periods=forecast_steps,
    freq="MS"
)

plt.figure(figsize=(9, 4))

plt.plot(
    last_12_df["Month"],
    last_12_df["Revenue"],
    label="Last 12 Months Revenue"
)

plt.plot(
    future_months,
    forecast,
    linestyle="--",
    label="Next Quarter Forecast"
)

plt.title("Quarterly Revenue Forecast (Using Last 12 Months)")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.legend()
plt.show()
