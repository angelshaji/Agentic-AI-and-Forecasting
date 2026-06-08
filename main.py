import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
import os
from dotenv import load_dotenv

    


print("KEY LOADED:", os.getenv("GROQ_API_KEY"))


# ===============================
# STEP 1: LOAD DATA
# ===============================
 
df = pd.read_csv("data/train.csv")
 
store_df = df[df["store"] == 1].copy()
store_df["date"] = pd.to_datetime(store_df["date"])
store_df["month"] = store_df["date"].dt.to_period("M")
 
monthly_df = (
    store_df
    .groupby("month")["sales"]
    .sum()
    .reset_index()
)
 
monthly_df.columns = ["Month", "Revenue"]
monthly_df["Month"] = monthly_df["Month"].dt.to_timestamp()
 
print("Last 12 months of revenue data:")
last_12_df = monthly_df.tail(12).reset_index(drop=True)
print(last_12_df)
 
# ===============================
# STEP 2: VALIDATION (Train/Test)
# ===============================
 
train = last_12_df["Revenue"][:9]
test  = last_12_df["Revenue"][9:]
 
model     = ARIMA(train, order=(1, 1, 0))
model_fit = model.fit()
 
predictions = model_fit.forecast(steps=3)
mae = mean_absolute_error(test, predictions)
print(f"\nModel MAE (Validation Error): {mae:.2f}")
print("MAE %:", mae / test.mean())
 
# ===============================
# STEP 3: FINAL MODEL
# ===============================
 
final_model = ARIMA(last_12_df["Revenue"], order=(1, 1, 0))
final_fit   = final_model.fit()
 
forecast_steps  = 3
forecast_object = final_fit.get_forecast(steps=forecast_steps)
conf_int        = forecast_object.conf_int()
forecast_values = forecast_object.predicted_mean

# Add forecast to dataframe (optional but powerful)
last_12_df["Forecast"] = None
last_12_df.loc[last_12_df.index[-forecast_steps:], "Forecast"] = forecast_values.values
 
print("\nForecasted revenue for next quarter:")
print(forecast_values)
 
# ===============================
# STEP 4: VISUALIZATION
# ===============================
 
last_month    = last_12_df["Month"].iloc[-1]
future_months = pd.date_range(
    start=last_month + pd.offsets.MonthBegin(),
    periods=forecast_steps,
    freq="MS"
)
 
plt.figure(figsize=(9, 4))
plt.plot(last_12_df["Month"], last_12_df["Revenue"], label="Last 12 Months Revenue")
plt.plot(future_months, forecast_values, linestyle="--", label="Forecast")
plt.fill_between(
    future_months,
    conf_int.iloc[:, 0],
    conf_int.iloc[:, 1],
    alpha=0.3,
    label="95% Confidence Interval"
)
plt.title("Quarterly Revenue Forecast with Confidence Intervals")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.legend()
plt.show(block=False)
plt.pause(3)
plt.close()
 
# ===============================
# STEP 5: INTERACTIVE AI AGENT
# ===============================

from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
import json


# Initialize Gemini

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)




@tool
def get_largest_months(n: int = 3) -> str:
    """Return the N months with the highest revenue."""
    idx = last_12_df["Revenue"].nlargest(n).index
    result = [{"month": last_12_df["Month"][i].strftime("%Y-%m"),
               "revenue": round(float(last_12_df["Revenue"][i]), 2)} for i in idx]
    return json.dumps(result)

@tool
def get_worst_months(n: int = 3) -> str:
    """Return the N months with the lowest revenue."""
    idx = last_12_df["Revenue"].nsmallest(n).index
    result = [{"month": last_12_df["Month"][i].strftime("%Y-%m"),
               "revenue": round(float(last_12_df["Revenue"][i]), 2)} for i in idx]
    return json.dumps(result)

@tool
def get_outliers(n: int = 3) -> str:
    """Detect outlier months using IQR method."""
    revenue = last_12_df["Revenue"]
    q1, q3  = revenue.quantile(0.25), revenue.quantile(0.75)
    iqr     = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask    = (revenue < lower) | (revenue > upper)
    out_df  = last_12_df[mask]
    if out_df.empty:
        idx    = (revenue - revenue.mean()).abs().nlargest(n).index
        out_df = last_12_df.loc[idx]
        note   = "No strict outliers. Showing most extreme values."
    else:
        note = "Outliers via IQR method."
        out_df = out_df.head(n)
    result = [{"month": row["Month"].strftime("%Y-%m"),
               "revenue": round(float(row["Revenue"]), 2)} for _, row in out_df.iterrows()]
    return json.dumps({"note": note, "outliers": result,
                       "bounds": {"lower": round(float(lower), 2), "upper": round(float(upper), 2)}})

@tool
def get_statistics() -> str:
    """Return summary statistics: mean, median, std, min, max, total."""
    r = last_12_df["Revenue"]
    return json.dumps({
        "mean":   round(float(r.mean()), 2),
        "median": round(float(r.median()), 2),
        "std":    round(float(r.std()), 2),
        "min":    round(float(r.min()), 2),
        "max":    round(float(r.max()), 2),
        "total":  round(float(r.sum()), 2)
    })

tools    = [get_largest_months, get_worst_months, get_outliers, get_statistics]
tool_map = {t.name: t for t in tools}
llm_with_tools = llm.bind_tools(tools)

SYSTEM = ("You are a revenue analyst. Use your tools to answer questions about "
          "the store's monthly revenue data. Always call a tool before answering.")

print("\n AI Agent Ready!")
print("Ask: 'biggest revenue month', 'outliers', 'worst months', 'stats summary'")
print("Type 'exit' to quit\n")

history = []

while True:
    query = input("👉 Your question: ").strip()
    if not query:
        continue
    if query.lower() == "exit":
        print("👋 Goodbye!")
        break

    history.append(HumanMessage(content=f"{SYSTEM}\n\nUser: {query}" if not history else query))
#**********************************************
# The agent will process the query, call tools as needed or answer directly without calling tools, and eventually provide an answer.
#****************************************************    
    while True:
        response = llm_with_tools.invoke(history)

        if not response.tool_calls:
            print(f"\n💡 Answer:\n{response.content}\n" + "-"*50 + "\n")
            history.append(response)
            break

        history.append(response)
        for tc in response.tool_calls:
            print(f"  [Tool: {tc['name']}]")
            result = tool_map[tc["name"]].invoke(tc["args"]) if tc["name"] in tool_map else "Unknown tool"
            history.append(ToolMessage(content=str(result), tool_call_id=tc.get("id", tc["name"])))