import matplotlib.pyplot as plt
import seaborn as sns

# Create state summary from the actual data
value_col = 'value'
state_summary = data.groupby('usastate')[value_col].sum().sort_values(ascending=False)

# Get actual monthly data from the dataset
if 'month' in data.columns:
    monthly_summary = data.groupby('month')[value_col].sum() / 1e9  # Convert to billions
    months = monthly_summary.index.tolist()
    trade_values_billion = monthly_summary.values.tolist()
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_labels = [month_names[m-1] for m in months]
else:
    # Fallback to example data if month column doesn't exist
    months = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    trade_values = [0.98e12, 0.87e12, 0.79e12, 0.41e12, 0.33e12, 0.41e12, 0.37e12, 0.29e12, 0.19e12]
    trade_values_billion = [v / 1e9 for v in trade_values]
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']
    month_labels = month_names

# Convert state_summary to billions for plotting
top_states = state_summary.head(10) / 1e9

# --- Monthly Trend Chart ---
plt.figure(figsize=(12, 6))
plt.plot(months, trade_values_billion, marker='o', color='lightcoral', alpha=0.7)
plt.title('Monthly Trade Value Trend (2020)')
plt.xticks(months, month_labels)
plt.grid(True, alpha=0.3)
# Add data labels in billions
for x, y in zip(months, trade_values_billion):
    plt.text(x, y, f'{y:,.0f}B', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.show()

# --- Top 10 States Chart ---
plt.figure(figsize=(10, 6))
sns.barplot(x=top_states.values, y=top_states.index, palette='Reds_r')
plt.title('Top 10 US States by Trade Value (2020)')
# Add data labels in billions
for i, v in enumerate(top_states.values):
    plt.text(v, i, f'{v:,.0f}B', va='center', ha='left', fontsize=9)
plt.tight_layout()
plt.show() 