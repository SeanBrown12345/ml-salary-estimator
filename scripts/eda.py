import os
import click
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
import numpy as np
import math
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.correlation_plot import corr_plot


@click.command()
@click.option('--input_path', type=str, required=True,
              help="Path to processed training data")
@click.option('--output_path', type=str, required=True,
              help="Path where figures and tables will be written, e.g., results/eda_")
def main(input_path, output_path):
    df = pd.read_csv(input_path)

    out_dir = os.path.dirname(output_path)
    if out_dir != "" and not os.path.exists(out_dir):
        os.makedirs(out_dir)


    numeric_col_list = [
    'age', 'education_num', 'capital_gain',
    'capital_loss', 'hours_per_week'
]
    
    zoom_ranges = {
        "capital_gain": (0, 50000),
        "capital_loss": (0, 5000),
        "hours_per_week": (0, 80)
    }
    
    # Academic-style titles
    academic_titles = {
        "age": "Age ",
        "education_num": "Years of Education",
        "capital_gain": "Capital Gain",
        "capital_loss": "Capital Loss",
        "hours_per_week": "Weekly Working Hours"
    }
    
    n_cols = 3
    n_rows = math.ceil(len(numeric_col_list) / n_cols)
    
    # smaller academic figure size
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.2 * n_cols, 3.2 * n_rows)
    )
    axes = axes.flatten()
    
    for i, feat in enumerate(numeric_col_list):
        ax = axes[i]
    
        df.groupby("income")[feat].plot.hist(
            bins=30, alpha=0.45, density=True,
            ax=ax,
            legend=False   # <-- REMOVE subplot legends
        )
    
        #ax.set_xlabel(feat, fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title(academic_titles[feat], fontsize=14, pad=8)
    
        if feat in zoom_ranges:
            ax.set_xlim(zoom_ranges[feat])
    
        ax.tick_params(labelsize=11)
    
    # Turn off unused subplot axes
    for j in range(len(numeric_col_list), n_rows * n_cols):
        axes[j].axis("off")
    
    # Global legend
    big_handles = [
        plt.Rectangle((0,0),1,1, color='#1f77b4', alpha=0.45),
        plt.Rectangle((0,0),1,1, color='#ff7f0e', alpha=0.45)
    ]
    big_labels = ["<=50K", ">50K"]
    
    fig.legend(
        big_handles, big_labels,
        title="Income Group",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=14,
        title_fontsize=16,
        frameon=False
    )
    fig.suptitle(
    "Distribution of Numeric Features by Income Group",
    fontsize=20,
    y=1.02)

    plt.tight_layout(rect=[0, 0, 0.95, 1])
    
    out_path = os.path.join(output_path, "numeric_feature_histograms.png")
    plt.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close()


  
    # 2. Spearman Correlation Including income_binary
    # -----------------------------------------------------

    # Copy df and convert income → binary
    df_corr = df.copy()
    df_corr["income"] = (df_corr["income"] == ">50K").astype(int)

    # Select all numeric columns including income_binary
    numeric_cols = df_corr.select_dtypes(include="number").columns.tolist()
    bubble_chart = corr_plot(df_corr,numeric_cols)
    bubble_chart.save(os.path.join(output_path, "correlation_bubble.png"))


    #3. Categorical Feature Proportion Plots
    cat_features = [
    "workclass",
    "education",
    "marital_status",
    "relationship",
    "race",
    "sex"]

    n_cols = 2
    n_rows = math.ceil(len(cat_features) / n_cols)
    
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(14, 12)
    )
    
    axes = axes.flatten()
    
    for i, feat in enumerate(cat_features):
        ax = axes[i]
    
        # Compute normalized category proportions
        temp = (
            df.groupby([feat, "income"])
            .size()
            .reset_index(name="count")
        )
        temp["proportion"] = temp.groupby(feat)["count"].transform(lambda x: x / x.sum())
    
        pivot_df = temp.pivot(index=feat, columns="income", values="proportion").fillna(0)
        pivot_df = pivot_df.sort_values("<=50K", ascending=False)
    
        pivot_df.plot(
            kind="bar",
            stacked=True,
            ax=ax,
            color=["#1f77b4", "#ff7f0e"],
            width=0.85,
            legend=False
        )
    
        # Title
        ax.set_title(feat.replace("_", " ").title(), fontsize=18, pad=10)
    
        # Y axis label
        ax.set_ylabel("Proportion", fontsize=16)
    
        # --- REMOVE x-axis label ---
        ax.set_xlabel("")   # << removed x-axis title
    
        # --- Make tick labels larger ---
        ax.tick_params(axis="x", labelsize=14, rotation=45)
        ax.tick_params(axis="y", labelsize=14)
    
    
    # Turn off unused axes
    for j in range(len(cat_features), n_rows * n_cols):
        axes[j].axis("off")
    
    # Global legend
    big_handles = [
        plt.Rectangle((0,0),1,1, color="#1f77b4", alpha=0.85),
        plt.Rectangle((0,0),1,1, color="#ff7f0e", alpha=0.85)
    ]
    big_labels = ["<=50K", ">50K"]
    
    fig.legend(
        big_handles,
        big_labels,
        title="Income Group",
        bbox_to_anchor=(1.02, 0.5),
        loc="center left",
        fontsize=16,
        title_fontsize=18,
        frameon=False
    )
    
    fig.suptitle("Categorical Feature Distributions by Income Group", fontsize=22, y=0.98)
    
    plt.tight_layout(rect=[0, 0, 0.92, 1])
    plt.savefig(os.path.join(output_path, "categorical_feature_bars.png"),
                dpi=250, bbox_inches="tight")
    plt.close()
if __name__ == '__main__':
    main()
