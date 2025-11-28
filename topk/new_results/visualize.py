import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import glob # Used to easily find all matching files

# Set Matplotlib and Seaborn style for publication-ready plots
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

# Define the SEES Attention-Confidence Gap threshold
SEES_GAP_THRESHOLD = 0.15

# --- Helper Function for Safe Correlation Calculation ---
def safe_pearsonr_r2(data, col1, col2):
    """Calculates Pearson R and R^2, returning NaN if N < 2 or data is constant."""
    n = len(data)
    # Require at least 3 points for a meaningful correlation plot
    if n < 3: 
        return pd.Series({'N': n, 'R': np.nan, 'R2': np.nan})
    else:
        # Check if either column is constant (which causes pearsonr to fail)
        if data[col1].nunique() <= 1 or data[col2].nunique() <= 1:
            return pd.Series({'N': n, 'R': np.nan, 'R2': np.nan})
        
        try:
            r, _ = pearsonr(data[col1], data[col2])
            return pd.Series({'N': n, 'R': r, 'R2': r**2})
        except Exception:
            # Catch other potential errors gracefully (e.g., all nans)
            return pd.Series({'N': n, 'R': np.nan, 'R2': np.nan})

def calculate_grouped_correlation(df, method_name):
    """Calculates and formats grouped correlation data."""
    grouped_corr = df.groupby('question_type').apply(
        lambda x: safe_pearsonr_r2(x, 'attention_entropy', 'token_confidence')
    ).reset_index()
    grouped_corr.columns = ['Question Type', 'N', 'R', 'R2']
    grouped_corr['Method'] = method_name
    return grouped_corr

# --- Data Loading and Processing ---
# List all files to be processed
# We include the 'full' file and use glob for all 'topk' files
files_to_process = ['analysis_results_final.csv'] + glob.glob('analysis_results_final_topk*.csv')

# Remove files that might be duplicates or unwanted (e.g., the original single 'topk' file)
files_to_process = [f for f in files_to_process if 'analysis_results_final_topk.csv' not in f]
files_to_process = sorted(list(set(files_to_process))) # Ensure unique and sort them

if not files_to_process:
    print("Error: No analysis files found. Ensure 'analysis_results_final.csv' and 'analysis_results_final_topk_*.csv' are available.")
    exit()

all_correlations = []
method_order = []

for file_path in files_to_process:
    try:
        df = pd.read_csv(file_path)
        
        if file_path == 'analysis_results_final.csv':
            method_name = 'Full Attention'
            method_order.insert(0, method_name) # Ensure Full Attention is first
        elif 'analysis_results_final_topk_' in file_path:
            k_value = file_path.split('_')[-1].replace('.csv', '')
            method_name = f'Top-K {k_value}'
            if method_name not in method_order:
                method_order.append(method_name)
        else:
            method_name = file_path # Should not happen with the current files
            if method_name not in method_order:
                method_order.append(method_name)
                
        print(f"Processing {file_path} as {method_name}...")
        
        # Calculate correlations and append to the list
        corr_df = calculate_grouped_correlation(df, method_name)
        all_correlations.append(corr_df)
        
    except FileNotFoundError:
        print(f"Warning: File not found: {file_path}. Skipping.")
    except Exception as e:
        print(f"Error processing {file_path}: {e}. Skipping.")

if not all_correlations:
    print("No data processed. Exiting.")
    exit()

# Combine all correlation dataframes
r_data = pd.concat(all_correlations, ignore_index=True).dropna(subset=['R', 'R2'])

# Filter out groups with insufficient samples (N < 3)
r_data = r_data[r_data['N'] >= 3]

# Define plot data
r2_plot_data = r_data.copy()
r_plot_data = r_data.copy()

# Sort the question types by the maximum R² across all methods for visual clarity
r2_plot_data_sorted = r2_plot_data.groupby('Question Type')['R2'].max().sort_values(ascending=False).index
r_plot_data['Question Type'] = pd.Categorical(r_plot_data['Question Type'], categories=r2_plot_data_sorted, ordered=True)
r2_plot_data['Question Type'] = pd.Categorical(r2_plot_data['Question Type'], categories=r2_plot_data_sorted, ordered=True)

# Define a consistent color palette and method order
method_order = ['Full Attention', 'Top-K 10', 'Top-K 25', 'Top-K 50', 'Top-K 100']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
color_map = dict(zip(method_order, colors))

r2_plot_data['Method'] = pd.Categorical(r2_plot_data['Method'], categories=method_order, ordered=True)
r_plot_data['Method'] = pd.Categorical(r_plot_data['Method'], categories=method_order, ordered=True)


# --- Plotting R² Comparison (The Attention-Confidence Gap) ---
plt.figure(figsize=(12, 7))
r2_plot = sns.barplot(
    data=r2_plot_data,
    x='Question Type',
    y='R2',
    hue='Method',
    palette=color_map,
    errorbar=None
)

# Add the Attention-Confidence Gap threshold line
r2_plot.axhline(
    SEES_GAP_THRESHOLD, 
    color='red', 
    linestyle='--', 
    linewidth=1.5, 
    label=f'SEES Gap Threshold ($R^2=0.15$)'
)

plt.title('R² Comparison: Full vs. Top-K Attention Entropy vs. Token Confidence', pad=15)
plt.ylabel(r'Coefficient of Determination ($R^2$)', fontsize=11)
plt.xlabel('Question Type', fontsize=11)
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 1.05)
plt.legend(title='Attention Method', loc='upper right')
plt.tight_layout()
plt.savefig('r2_comparison_all_methods.png', dpi=300)
plt.show()

# --- Plotting R Comparison (The Direction of Correlation) ---
plt.figure(figsize=(12, 7))
r_plot = sns.barplot(
    data=r_plot_data,
    x='Question Type',
    y='R',
    hue='Method',
    palette=color_map,
    errorbar=None
)

# Add line for zero correlation
r_plot.axhline(0, color='black', linestyle='-', linewidth=0.8)

plt.title('Pearson R Comparison: Focused (Negative) vs. Dispersed (Positive) Attention', pad=15)
plt.ylabel(r'Pearson Coefficient ($R$)', fontsize=11)
plt.xlabel('Question Type', fontsize=11)
plt.xticks(rotation=45, ha='right')
plt.ylim(-1.05, 1.05)
plt.legend(title='Attention Method', loc='lower right')
plt.tight_layout()
plt.savefig('r_comparison_all_methods.png', dpi=300)
plt.show()

print("\nVisualization script complete. Two files, 'r2_comparison_all_methods.png' and 'r_comparison_all_methods.png', have been generated.")
print("The R² plot now compares all Top-K variants against the Full Attention method.")
print("The R plot now compares the sign and magnitude of the Pearson R for all methods.")