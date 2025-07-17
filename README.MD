## TransBorder Freight Data Analysis (2020)
Overview
This project analyzes the 2020 US TransBorder Freight Data using the CRISP-DM (Cross-Industry Standard Process for Data Mining) methodology. The goal is to identify inefficiencies, uncover patterns, and propose actionable recommendations to improve the efficiency, sustainability, and cost-effectiveness of cross-border freight operations between the US, Canada, and Mexico.

## Table of Contents
CRISP-DM Framework Implementation
Analytical Questions
Data Preprocessing
Visualizations & Insights
Conclusions & Actionable Recommendations


## CRISP-DM Framework Implementation
1. Business Understanding
Objective: Enhance the efficiency, sustainability, and safety of North American trans-border freight.
Key Focus: Identify inefficiencies, recognize trade patterns, and propose actionable solutions to optimize performance and minimize costs and environmental impact.

2. Data Understanding
Data Source: Bureau of Transportation Statistics (BTS) 2020 TransBorder Freight Data.
Data Structure: Monthly CSV files covering trade value, weight, mode of transportation, port/border, commodity, and more.
Initial Exploration: Inspected for missing values, outliers, and consistency across months, ports, and modes.

3. Data Preparation
Loading: All 2020 CSV files were loaded and concatenated into a single DataFrame.

Cleaning:
Standardized column names.
Converted value and weight columns to numeric types.
Filled missing numeric values with 0 and categorical values with 'Unknown'.
Removed or flagged duplicate records.

Feature Engineering:
Derived new features such as trade balance (export - import) and value per ton.

4. Modeling & Analysis
Aggregated data by state, port, mode, commodity, and month.
Identified inefficiencies such as zero-weight, high-value shipments and underutilized routes.
Analyzed seasonal and modal trends.

5. Evaluation
Visualized key findings using bar charts, line plots, and boxplots.
Assessed the impact of COVID-19 and other anomalies on trade flows.

6. Deployment
Documented insights and recommendations.

## Exported summary tables and figures for reporting.

Analytical Questions
Which U.S. states have the highest and lowest trans-border trade values with Canada and Mexico?

What are the top ports of entry/exit for trans-border freight, and how do they compare in terms of trade volume and value?

Which modes of transportation dominate trans-border freight, and how does their usage vary by region or commodity?

What are the leading commodities traded across U.S. borders, and how do their trade values and volumes change over time?

How do trade flows and values fluctuate throughout the year? Are there clear seasonal patterns or anomalies?

Where are the inefficiencies in freight flows, such as shipments with high value but zero or low reported weight, or underutilized routes?

What actionable recommendations can be made to improve efficiency, sustainability, and cost-effectiveness in trans-border freight operations?

Data Preprocessing
File Discovery: Used Python’s glob and os libraries to recursively find and load all relevant CSV files for 2020.

Data Cleaning:
Standardized column names for consistency.
Converted monetary and weight columns to numeric, removing symbols and commas.
Filled missing values appropriately.
Checked for and handled duplicate records.

Feature Engineering:
Created new columns for trade balance and value per ton where possible.

Validation:
Checked for missing months, ports, and modes to ensure data completeness.

Visualized distributions to spot outliers and anomalies.
Visualizations & Insights

Top States by Trade Value:
Bar charts highlighted Texas, Michigan, and California as the leading states in cross-border trade.

Top Ports:
Laredo and Detroit emerged as the most significant ports by trade value.

Trade by Mode:
Trucking dominated, followed by rail and vessel, with clear visual representation of modal shares.

Commodity Analysis:
Vehicles, mineral fuels, and machinery were the top commodities traded.

Monthly Trends:
Line plots revealed seasonal patterns and the impact of COVID-19, with a noticeable dip in early 2020 and recovery later in the year.

Inefficiency Analysis:
Boxplots and summary tables identified shipments with zero or low weight but high value, suggesting potential reporting or operational inefficiencies.

Congestion & Under-utilization:
Analysis of shipment weights and values by route and mode revealed congested and underutilized corridors.

## Conclusions & Actionable Recommendations
Key Findings
Texas, Michigan, and California are the top states for cross-border trade.
Laredo and Detroit are the busiest ports.
Trucking is the dominant mode, but also the largest contributor to carbon emissions.
Significant inefficiencies exist, such as high-value shipments with zero reported weight and underutilized routes.
Seasonal patterns and external events (e.g., COVID-19) significantly impact trade flows.

## **Actionable Recommendations**
Invest in rail infrastructure and underutilized routes to reduce congestion and environmental impact.

Encourage greener fuel alternatives and carbon reduction strategies, especially for trucking.

Optimize rail and vessel transport for cost-effective and sustainable trade.

Implement AI-driven logistics for better freight management and real-time monitoring.

Harmonize trade policies to minimize delays and inefficiencies.
Improve data collection and reporting to reduce anomalies such as zero-weight, high-value shipments.

## **Conclusion**

References
Bureau of Transportation Statistics (BTS)
TransBorder Freight Data Analysis Example
This project demonstrates the power of data-driven analysis in optimizing cross-border freight operations, supporting both economic and environmental goals.



