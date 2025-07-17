# BTS TransBorder Freight Data Analysis 2020rehensive Analysis using CRISP-DM Methodology

# Skills Demonstrated:
# - Data Science: CRISP-DM methodology, statistical analysis, data visualization
# - Programming: Python, pandas, matplotlib, seaborn, numpy
# - Business Intelligence: Trade flow analysis, transportation efficiency, seasonal patterns
# - Data Engineering: Data preprocessing, cleaning, aggregation
# - Visualization: Interactive charts, statistical plots, geographic analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8)
sns.set_palette("husl")

class TransBorderAnalyzer:
    def __init__(self, data_path="2020        Initialize the TransBorder Freight Data Analyzer
        
        Parameters:
        data_path (str): Path to the data directory containing monthly folders
      
        self.data_path = data_path
        self.data = None
        self.monthly_data = {}
        
    def load_data(self):
           CRISP-DM Stage 1: Business Understanding & Data Understanding
        Load and understand the structure of TransBorder Freight Data
               print("=== CRISP-DM Stage 1: Business Understanding & Data Understanding ===)    print(Loading BTS TransBorder Freight Data for 2020")
        
        all_data = []
        
        # Load data for each month (January to December)
        for month in range(1, 13):
            month_name = f"[object Object]month:02d}  # 01, 02, etc.
            month_folder = os.path.join(self.data_path, month_name)
            
            if os.path.exists(month_folder):
                # Search for CSV files recursively in the month folder
                csv_files = glob.glob(os.path.join(month_folder, "**/*.csv), recursive=True)
                
                if csv_files:
                    for file_path in csv_files:
                        try:
                            df = pd.read_csv(file_path)
                            dfMonth                   df[Month_Name'] = pd.to_datetime(f220-{month:2}-1).strftime('%B')                   all_data.append(df)
                            print(fLoaded {file_path}")
                        except Exception as e:
                            print(f"Error loading {file_path}: {e})              else:
                    print(f"No CSV files found in {month_folder}")
            else:
                print(f"Month folder {month_folder} not found")
        
        if all_data:
            self.data = pd.concat(all_data, ignore_index=True)
            print(f"\nData loaded successfully!")
            print(f"Total records: {len(self.data):,}")
            print(f"Columns: {list(self.data.columns)}")
            print(fDate range: {self.data['Month_Name].min()} to {self.data['Month_Name'].max()}")
            
            # Display basic statistics
            print("\n=== Data Overview ===")
            print(self.data.info())
            print(n=== Sample Data ===")
            print(self.data.head())
        else:
            print("No data files found!")
            return False
        
        returntrue  
    def prepare_data(self):
           CRISP-DM Stage 2: Data Preparation
        Clean, transform, and prepare data for analysis
       
        print("\n=== CRISP-DM Stage 2: Data Preparation ===")
        
        if self.data is None:
            print("No data loaded. Please run load_data() first.")
            return False
        
        # Create a copy to avoid modifying original data
        df = self.data.copy()
        
        # Handle missing values
        print("Handling missing values...)
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(0)
        
        # Convert value columns to numeric, removing any currency symbols
        value_columns = [col for col in df.columns if 'value' in col.lower() or 'dollars in col.lower()]
        for col in value_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace('$,).str.replace(,, ), errors='coerce)                df[col] = df[col].fillna(0)
        
        # Convert weight columns to numeric
        weight_columns = [col for col in df.columns if 'weight' in col.lower() or 'tons in col.lower()]
        for col in weight_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(,, ), errors='coerce)                df[col] = df[col].fillna(0)
        
        # Standardize column names
        df.columns = df.columns.str.lower().str.replace(', )
        
        # Create derived features
        print("Creating derived features...")
        
        # Calculate trade balance if import/export data available
        ifimport_value' in df.columns andexport_value' in df.columns:
            df[trade_balance'] = df['export_value'] - df['import_value']
        
        # Calculate value per ton if both value and weight available
        value_cols = [col for col in df.columns if value in col and perot in col]
        weight_cols = [col for col in df.columns if weight in col or 'tons' in col]
        
        for val_col in value_cols:
            for wt_col in weight_cols:
                if val_col != wt_col:
                    col_name = f{val_col}_per_ton"
                    df[col_name] = np.where(df[wt_col] > 0, df[val_col] / df[wt_col], 0)
        
        self.data = df
        print("Data preparation completed!)
        print(f"Final dataset shape: {self.data.shape}")
        
        returntrue  
    def analyze_trade_flows(self):
           CRISP-DM Stage 3: Modeling & Analysis
        Analyze trade flows and patterns
       
        print("\n=== CRISP-DM Stage 3: Modeling & Analysis ===)      print("Analyzing trade flows...")
        
        # Create plots directory
        os.makedirs('plots, exist_ok=True)
        
        # 1. Monthly Trade Volume Analysis
        print("1. Monthly Trade Volume Analysis")
        
        # Group by month and calculate totals
        monthly_summary = self.data.groupby(['month,month_name']).agg([object Object]            col: 'sum' for col in self.data.select_dtypes(include=[np.number]).columns
        }).reset_index()
        
        # Sort by month
        monthly_summary = monthly_summary.sort_values(month)
        
        # Plot monthly trends
        plt.figure(figsize=(15)
        
        # Find value columns for plotting
        value_cols = [col for col in monthly_summary.columns if value' in col and col not in ['month, _name']]
        
        for i, col in enumerate(value_cols[:4]):  # Plot first 4 value columns
            plt.subplot(2, 2, i+1          plt.plot(monthly_summary['month_name'], monthly_summary[col], marker='o', linewidth=2, markersize=8         plt.title(f'Monthly[object Object]col.replace("_, ).title()}', fontsize=14, fontweight='bold)        plt.xlabel('Month)        plt.ylabel('Value ($))
            plt.xticks(rotation=45          plt.grid(True, alpha=0.3      
            # Add value labels on points
            for x, y in zip(range(len(monthly_summary)), monthly_summary[col]):
                plt.annotate(f${y:,.0}', (x, y), textcoords="offset points", xytext=(0,10), ha=center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('plots/monthly_trade_trends.png', dpi=300bbox_inches='tight')
        plt.show()
        
        #2ransportation Mode Analysis
        print(2. Transportation Mode Analysis")
        
        # Find mode-related columns
        mode_cols = [col for col in self.data.columns if any(mode in col.lower() for mode in [mode, transport', truck',rail', 'air', 'vessel'])]
        
        if mode_cols:
            plt.figure(figsize=(15, 10))
            for i, col in enumerate(mode_cols[:4]):
                if col in self.data.columns:
                    mode_data = self.data.groupby(col).agg({
                        val_col:sum' for val_col in value_cols if val_col in self.data.columns
                    }).reset_index()
                    
                    if len(mode_data) > 1 # Only plot if we have multiple modes
                        plt.subplot(2, 2, i+1)
                        
                        # Get the first value column for plotting
                        plot_col = [col for col in mode_data.columns if value' in col]0 if any(value' in col for col in mode_data.columns) else mode_data.columns[1]
                        
                        plt.bar(range(len(mode_data)), mode_data[plot_col])
                        plt.title(f{col.replace(_le()} Distribution', fontsize=14, fontweight='bold')
                        plt.xlabel(col.replace('_', ' ').title())
                        plt.ylabel('Value ($)')
                        plt.xticks(range(len(mode_data)), mode_data[col], rotation=45)
                        
                        # Add value labels on bars
                        for j, v in enumerate(mode_data[plot_col]):
                            plt.text(j, v, f'${v:,.0, ha='center', va=bottom', fontsize=8)
            
            plt.tight_layout()
            plt.savefig('plots/transportation_mode_analysis.png', dpi=300bbox_inches='tight)          plt.show()
        
        # 3. Seasonal Pattern Analysis
        print("3. Seasonal Pattern Analysis")
        
        # Create seasonal categories
        self.data[season'] = pd.cut(self.data['month'], 
                                    bins=[0                   labels=['Winter',Spring, Summer', Fall)
        
        seasonal_analysis = self.data.groupby('season').agg([object Object]            col: 'sum' for col in self.data.select_dtypes(include=[np.number]).columns if month' not in col
        }).reset_index()
        
        plt.figure(figsize=(15)
        
        for i, col in enumerate(value_cols[:4]):
            plt.subplot(2, 2, i+1)
            colors =#FF6B6, #4ECDC4, #45B7D1', '#96CEB4]
            plt.pie(seasonal_analysis[col], labels=seasonal_analysis[season], autopct=%1.1f%%, colors=colors)
            plt.title(f'Seasonal Distribution -[object Object]col.replace("_, ).title()}', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('plots/seasonal_patterns.png', dpi=300bbox_inches='tight')
        plt.show()
        
        # 4. Port Performance Analysis
        print("4. Port Performance Analysis")
        
        # Find port-related columns
        port_cols = [col for col in self.data.columns if 'port' in col.lower() or 'border in col.lower()]
        
        if port_cols:
            plt.figure(figsize=(15, 10))
            for i, col in enumerate(port_cols[:4]):
                if col in self.data.columns:
                    port_data = self.data.groupby(col).agg({
                        val_col:sum' for val_col in value_cols if val_col in self.data.columns
                    }).reset_index()
                    
                    # Get top 10 ports
                    plot_col = [col for col in port_data.columns if value' in col]0 if any(value' in col for col in port_data.columns) else port_data.columns[1]
                    top_ports = port_data.nlargest(10, plot_col)
                    
                    plt.subplot(2, 2, i+1)
                    plt.barh(range(len(top_ports)), top_ports[plot_col])
                    plt.title(f'Top 10[object Object]col.replace("_, ).title()}', fontsize=14, fontweight='bold')
                    plt.xlabel('Value ($)')
                    plt.yticks(range(len(top_ports)), top_ports[col])
                    
                    # Add value labels
                    for j, v in enumerate(top_ports[plot_col]):
                        plt.text(v, j, f'${v:,.0 va=center', fontsize=8)
            
            plt.tight_layout()
            plt.savefig('plots/port_performance.png', dpi=300bbox_inches='tight)          plt.show()
    
    def evaluate_and_visualize(self):
           CRISP-DM Stage 4: Evaluation & Visualization
        Create comprehensive visualizations and evaluate findings
       
        print("\n=== CRISP-DM Stage 4: Evaluation & Visualization ===")
        
        # 1. Correlation Analysis
        print("1. Correlation Analysis")
        
        numeric_data = self.data.select_dtypes(include=[np.number])
        correlation_matrix = numeric_data.corr()
        
        plt.figure(figsize=(12,10
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap=coolwarm', center=0,
                   square=True, linewidths=0.5 cbar_kws={"shrink": .8})
        plt.title('Correlation Matrix of Numeric Variables', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(plots/correlation_matrix.png', dpi=300bbox_inches='tight')
        plt.show()
        
        # 2. Summary Statistics Dashboard
        print("2. Summary Statistics Dashboard")
        
        # Calculate key metrics
        summary_stats = {}
        
        # Total values
        value_cols = [col for col in self.data.columns if value' in col and col not in ['month, th_name']]
        for col in value_cols:
            if col in self.data.columns:
                summary_stats[f'Total_{col}] = self.data[col].sum()
                summary_stats[f'Avg_{col}] = self.data[col].mean()
                summary_stats[f'Max_{col}] = self.data[col].max()
                summary_stats[f'Min_{col}] = self.data[col].min()
        
        # Create summary visualization
        fig, axes = plt.subplots(2, 2figsize=(15)
        
        # Total values bar chart
        total_values = {k: v for k, v in summary_stats.items() if k.startswith(Total_')}
        if total_values:
            axes[0, 0].bar(range(len(total_values)), list(total_values.values()))
            axes[0, 0].set_title('Total Values by Category', fontweight='bold')
            axes[0, 0].set_xticks(range(len(total_values)))
            axes[0, 0.set_xticklabels([k.replace('Total_',  k in total_values.keys()], rotation=45)
            axes[0.set_ylabel('Value ($)')
            
            # Add value labels
            for i, v in enumerate(total_values.values()):
                axes[0, 0].text(i, v, f'${v:,.0, ha='center', va=bottom', fontsize=8)
        
        # Average values
        avg_values = {k: v for k, v in summary_stats.items() if k.startswith('Avg_)}     if avg_values:
            axes[0,1.bar(range(len(avg_values)), list(avg_values.values()))
            axes0,1set_title(AverageValues by Category', fontweight='bold')
            axes[0, 1].set_xticks(range(len(avg_values)))
            axes[0, 1.set_xticklabels([k.replace('Avg_', '') for k in avg_values.keys()], rotation=45)
            axes[0.set_ylabel('Value ($)')
        
        # Monthly distribution
        monthly_totals = self.data.groupby('month_name').agg([object Object]            col: 'sum for col in value_cols if col in self.data.columns
        }).reset_index()
        
        if len(monthly_totals) > 0:
            plot_col = [col for col in monthly_totals.columns if value' in col]0 if any(value' in col for col in monthly_totals.columns) else monthly_totals.columns[1]
            axes[1, 0ot(range(len(monthly_totals)), monthly_totals[plot_col], marker=odth=2)
            axes1,0set_title('Monthly Trend', fontweight='bold')
            axes[1.set_xlabel('Month')
            axes[1.set_ylabel('Value ($)')
            axes[1, 0].set_xticks(range(len(monthly_totals)))
            axes[1, 0].set_xticklabels(monthly_totals['month_name'], rotation=45)
            axes[1, 0].grid(True, alpha=0.3)
        
        # Data distribution histogram
        if value_cols:
            plot_col = value_cols[0]
            axes[1, 1ist(self.data[plot_col], bins=30pha=0.7, edgecolor='black')
            axes1tle(fDistribution of[object Object]plot_col.replace("_, ).title()}', fontweight='bold')
            axes[1.set_xlabel('Value ($)')
            axes[1_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('plots/summary_dashboard.png', dpi=300bbox_inches='tight')
        plt.show()
        
        #3summary statistics
        print("3. Exporting Summary Statistics")
        
        summary_df = pd.DataFrame(list(summary_stats.items()), columns=['Metric',Value])
        summary_df.to_csv('plots/summary_statistics.csv', index=False)
        print("Summary statistics saved to plots/summary_statistics.csv")
    
    def deploy_and_conclude(self):
           CRISP-DM Stage 5: Deployment & Conclusions
        Generate final report and conclusions
       
        print("\n=== CRISP-DM Stage 5: Deployment & Conclusions ===")
        
        # Generate comprehensive report
        report =     report.append("BTS TransBorder Freight Data Analysis Report 2020     report.append("=" * 50     report.append("")
        
        # Executive Summary
        report.append(EXECUTIVE SUMMARY")
        report.append("-" * 20     report.append(fAnalysis Period: 2020     report.append(f"Total Records Analyzed: {len(self.data):,}")
        report.append(f"Data Quality: {self.data.isnull().sum().sum()} missing values")
        report.append("")
        
        # Key Findings
        report.append("KEY FINDINGS")
        report.append("-" * 15)
        
        # Calculate key metrics
        value_cols = [col for col in self.data.columns if value' in col and col not in ['month, _name']]
        
        if value_cols:
            total_values = [object Object]           for col in value_cols:
                if col in self.data.columns:
                    total_values[col] = self.data[col].sum()
            
            # Find highest and lowest months
            monthly_totals = self.data.groupby('month_name').agg({
                col: 'sum for col in value_cols if col in self.data.columns
            }).reset_index()
            
            if len(monthly_totals) > 0              plot_col = [col for col in monthly_totals.columns if value' in col]0 if any(value' in col for col in monthly_totals.columns) else monthly_totals.columns[1               max_month = monthly_totals.loc[monthly_totals[plot_col].idxmax()]
                min_month = monthly_totals.loc[monthly_totals[plot_col].idxmin()]
                
                report.append(f"• Peak Month: {max_month[month_name']} (${max_month[plot_col]:,.0f}))            report.append(f•Lowest Month: {min_month[month_name']} (${min_month[plot_col]:,.0f}))            report.append(f• Total Value: ${total_values.get(plot_col, 0):,.0f}")
        
        report.append("")
        
        # Methodology
        report.append("METHODOLOGY")
        report.append("-" * 12     report.append("This analysis follows the CRISP-DM methodology:")
        report.append("1. Business Understanding: Defined objectives and requirements")
        report.append("2. Data Understanding: Explored and characterized the data")
        report.append("3. Data Preparation: Cleaned, transformed, and prepared data")
        report.append("4. Modeling: Applied statistical analysis and visualization")
        report.append("5. Evaluation: Assessed results and generated insights")
        report.append("6. Deployment: Created actionable recommendations")
        report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 15     report.append("• Monitor seasonal patterns for capacity planning")
        report.append("• Analyze transportation mode efficiency")
        report.append("• Track port performance metrics")
        report.append("• Consider trade balance implications")
        report.append("• Implement real-time monitoring systems")
        report.append("")
        
        # Save report
        with open('plots/analysis_report.txt', 'w') as f:
            f.write('\n'.join(report))
        
        print("Analysis report saved to plots/analysis_report.txt")
        print("\nAnalysis completed successfully!)        print("All visualizations saved in theplots' directory")
        
        return '\n'.join(report)

def main():
    """
    Main function to run the complete analysis
      print("BTS TransBorder Freight Data Analysis 2020)
    print("Using CRISP-DM Methodology)
    print(= * 50) 
    # Initialize analyzer
    analyzer = TransBorderAnalyzer(2020    # Run complete analysis pipeline
    if analyzer.load_data():
        if analyzer.prepare_data():
            analyzer.analyze_trade_flows()
            analyzer.evaluate_and_visualize()
            report = analyzer.deploy_and_conclude()
            print("\n" + report)
        else:
            print("Data preparation failed!")
    else:
        print("Data loading failed!)if __name__ == "__main__":
    main() 