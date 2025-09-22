import os
import glob
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

# -------------------------------
# App Config
# -------------------------------
st.set_page_config(page_title="TransBorder Freight Analysis", layout="wide")

# -------------------------------
# Data Loading Utilities
# -------------------------------
@st.cache_data(show_spinner=True)
def load_all_data(base_dir: str) -> pd.DataFrame:
	csv_patterns = [
		os.path.join(base_dir, '2020', '**', '*.csv'),
		os.path.join(base_dir, '2021', '**', '*.csv'),
		os.path.join(base_dir, '2022', '**', '*.csv'),
		os.path.join(base_dir, '2023', '**', '*.csv'),
		os.path.join(base_dir, '2024', '**', '*.csv'),
	]
	csv_files = []
	for pattern in csv_patterns:
		found = glob.glob(pattern, recursive=True)
		csv_files.extend(found)
		# Debug: show what files were found for each pattern
		if found:
			print(f"Pattern {pattern} found {len(found)} files")
	
	print(f"Total CSV files found: {len(csv_files)}")

	dfs = []
	for file in csv_files:
		try:
			df = pd.read_csv(file, low_memory=False)
			df['source_file'] = os.path.relpath(file, base_dir)
			dfs.append(df)
		except Exception as e:
			st.warning(f"Error loading {file}: {e}")

	if not dfs:
		return pd.DataFrame()

	data = pd.concat(dfs, ignore_index=True)

	# Standardize columns
	data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_')

	# Ensure expected columns exist
	expected_numeric_like = ['value', 'shipwt', 'freight_charges']
	for col in expected_numeric_like:
		if col in data.columns:
			if col == 'value':
				data[col] = pd.to_numeric(
					data[col].astype(str).str.replace(r'[$,]', '', regex=True), errors='coerce'
				)
			else:
				data[col] = pd.to_numeric(
					data[col].astype(str).str.replace(',', '', regex=True), errors='coerce'
				)

	# Missing values
	num_cols = data.select_dtypes(include=[np.number]).columns
	cat_cols = data.select_dtypes(include=['object']).columns
	if len(num_cols) > 0:
		data[num_cols] = data[num_cols].fillna(0)
	if len(cat_cols) > 0:
		data[cat_cols] = data[cat_cols].fillna('Unknown')

	# Year and month fallback if not present
	if 'year' not in data.columns:
		data['year'] = data['source_file'].str.extract(r'(20\d{2})').fillna('Unknown')
	if 'month' not in data.columns:
		data['month'] = 0

	return data

# -------------------------------
# Startup + Diagnostics
# -------------------------------
st.title("TransBorder Freight Analysis")

try:
	# Use current directory since year folders are in the same directory as the script
	base_dir = os.path.dirname(__file__)
	env_info = {
		"__file__": __file__,
		"base_dir": base_dir,
		"cwd": os.getcwd(),
	}
	data = load_all_data(base_dir)
except Exception as e:
	st.error("Failed during startup or data loading.")
	st.exception(e)
	st.stop()

with st.expander("Diagnostics"):
	st.write(env_info)
	# Count CSVs discovered
	counts = {}
	for y in ['2020', '2021', '2022', '2023', '2024']:
		p = os.path.join(base_dir, y)
		counts[y] = sum(1 for _ in glob.iglob(os.path.join(p, '**', '*.csv'), recursive=True)) if os.path.exists(p) else 0
	st.write({"csv_counts": counts})
	st.write({"rows": len(data), "cols": len(data.columns) if not data.empty else 0})
	if not data.empty:
		st.write("Head:")
		st.dataframe(data.head(5))
		st.write("Columns:")
		st.code(", ".join(data.columns.tolist())[:2000])

if data.empty:
	st.error("No CSV files found. Ensure the year folders exist and contain CSVs.")
	st.stop()

# Basic column name mappings from notebook
state_col = 'usastate' if 'usastate' in data.columns else None
mode_col = 'disagmot' if 'disagmot' in data.columns else None
value_col = 'value' if 'value' in data.columns else None
weight_col = 'shipwt' if 'shipwt' in data.columns else None
commodity_col = next((c for c in data.columns if 'commodity' in c), None)

# -------------------------------
# Sidebar Controls
# -------------------------------
with st.sidebar:
	st.header("Filters")
	years = sorted([y for y in data['year'].unique().tolist() if str(y) != 'Unknown']) if 'year' in data.columns else []
	selected_years = st.multiselect("Year", years, default=years[:1] if years else [])

	modes = sorted(data[mode_col].unique().tolist()) if mode_col else []
	selected_modes = st.multiselect("Mode", modes, default=modes)

	states = sorted(data[state_col].unique().tolist()) if state_col else []
	selected_states = st.multiselect("State", states, default=states[:10] if len(states) > 10 else states)

# Apply filters safely
filtered = data
try:
	if selected_years and 'year' in filtered.columns:
		filtered = filtered[filtered['year'].astype(str).isin([str(y) for y in selected_years])]
	if selected_modes and mode_col:
		filtered = filtered[filtered[mode_col].isin(selected_modes)]
	if selected_states and state_col:
		filtered = filtered[filtered[state_col].isin(selected_states)]
except Exception as e:
	st.error("Error applying filters.")
	st.exception(e)

# -------------------------------
# KPI Cards
# -------------------------------
try:
	col1, col2, col3, col4 = st.columns(4)
	with col1:
		total_trade = float(filtered[value_col].sum()) if value_col and value_col in filtered.columns else 0.0
		st.metric("Total Trade Value", f"${total_trade:,.0f}")
	with col2:
		total_weight = float(filtered[weight_col].sum()) if weight_col and weight_col in filtered.columns else 0.0
		st.metric("Total Weight", f"{total_weight:,.0f}")
	with col3:
		records = int(len(filtered))
		st.metric("Records", f"{records:,}")
	with col4:
		zero_weight = int((filtered[weight_col] == 0).sum()) if weight_col and weight_col in filtered.columns else 0
		st.metric("Zero-weight Shipments", f"{zero_weight:,}")
except Exception as e:
	st.error("Error computing KPIs.")
	st.exception(e)

st.markdown("---")

# -------------------------------
# Charts with guards
# -------------------------------
try:
	# Monthly Trend
	if 'month' in filtered.columns and value_col and value_col in filtered.columns:
		monthly = (
			filtered.groupby('month')[value_col]
			.sum()
			.reset_index()
			.sort_values('month')
		)
		monthly['value_billion'] = monthly[value_col] / 1e9
		fig = px.line(
			monthly,
			x='month', y='value_billion', markers=True,
			title='Monthly Trade Value (Billions)'
		)
		st.plotly_chart(fig, use_container_width=True)

	# Top States
	if state_col and value_col and state_col in filtered.columns and value_col in filtered.columns:
		state_summary = (
			filtered.groupby(state_col)[value_col]
			.sum()
			.sort_values(ascending=False)
			.head(15)
			.reset_index()
		)
		state_summary['value_billion'] = state_summary[value_col] / 1e9
		fig = px.bar(
			state_summary, x='value_billion', y=state_col, orientation='h',
			title='Top States by Trade Value (Billions)'
		)
		st.plotly_chart(fig, use_container_width=True)

	# Mode Distribution
	if mode_col and value_col and mode_col in filtered.columns and value_col in filtered.columns:
		mode_summary = (
			filtered.groupby(mode_col)[value_col]
			.sum()
			.reset_index()
		)
		mode_summary['value_billion'] = mode_summary[value_col] / 1e9
		fig = px.bar(
			mode_summary, x=mode_col, y='value_billion',
			title='Trade Value by Transportation Mode (Billions)'
		)
		st.plotly_chart(fig, use_container_width=True)

	# Top Commodities
	if commodity_col and value_col and commodity_col in filtered.columns and value_col in filtered.columns:
		commodity_summary = (
			filtered.groupby(commodity_col)[value_col]
			.sum()
			.sort_values(ascending=False)
			.head(15)
			.reset_index()
		)
		commodity_summary['value_billion'] = commodity_summary[value_col] / 1e9
		fig = px.bar(
			commodity_summary, x='value_billion', y=commodity_col, orientation='h',
			title='Top Commodities by Trade Value (Billions)'
		)
		st.plotly_chart(fig, use_container_width=True)
except Exception as e:
	st.error("Error rendering charts.")
	st.exception(e)

with st.expander("Notes"):
	st.write(
		"""
		- Column names are standardized to lower_snake_case.
		- Numeric conversions are applied to `value`, `shipwt`, and `freight_charges` where present.
		- Year is inferred from `source_file` when missing.
		- Use the sidebar to filter the dataset; KPIs and charts respond to filters.
		"""
	)
