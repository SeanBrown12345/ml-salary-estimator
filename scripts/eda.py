import os
import click
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
import math


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
    df_corr["income_binary"] = (df_corr["income"] == ">50K").astype(int)

    # Select all numeric columns including income_binary
    numeric_cols = df_corr.select_dtypes(include="number").columns.tolist()

    # Compute correlation matrix
    corr_df = (
        df_corr[numeric_cols]
        .corr(method='spearman')
        .stack()
        .reset_index()
    )
    corr_df.columns = ['Feature 1', 'Feature 2', 'Correlation']
    corr_df['Absolute Correlation'] = corr_df['Correlation'].abs()

    # ----- Save correlation bubble chart -----
    bubble_chart = (
    alt.Chart(corr_df)
    .mark_circle()
    .encode(
        x=alt.X(
            'Feature 1:N',
            title='Feature 1',
            axis=alt.Axis(labelAngle=-45, labelFontSize=16, titleFontSize=20)
        ),
        y=alt.Y(
            'Feature 2:N',
            title='Feature 2',
            axis=alt.Axis(labelFontSize=16, titleFontSize=20)
        ),
        size=alt.Size(
            'Absolute Correlation:Q',
            scale=alt.Scale(domain=(0, 1)),
            legend=alt.Legend(
                title="Absolute Correlation",
                titleFontSize=18,
                labelFontSize=16
            )
        ),
        color=alt.Color(
            'Correlation:Q',
            scale=alt.Scale(scheme='blueorange', domain=(-1, 1)),
            legend=alt.Legend(
                title="Correlation",
                titleFontSize=18,
                labelFontSize=16
            )
        ),
        tooltip=[
            'Feature 1',
            'Feature 2',
            alt.Tooltip('Correlation:Q', format='.3f'),
            alt.Tooltip('Absolute Correlation:Q', format='.3f')
        ]
    )
    .properties(
        width=580,    
        height=580,
        title=alt.TitleParams(
            "Spearman Correlation Bubble Chart",
            fontSize=30,
            anchor="middle"
        )
    )
    .configure_view(
        strokeWidth=0 
    )
    .configure_legend(
        titleFontSize=18,
        labelFontSize=16
    ))

    bubble_chart.save(os.path.join(output_path, "correlation_bubble.png"))


if __name__ == '__main__':
    main()
