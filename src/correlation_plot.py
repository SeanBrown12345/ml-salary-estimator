import altair as alt
import numpy as np
import pandas as pd

def corr_plot(dataframe, feature_list):
    """
    Create a correlation bubble chart for selected features.

    This function computes the pairwise Pearson correlation coefficients
    among the specified numeric features and visualizes them using an
    Altair bubble chart. 

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Input dataframe containing the features to be analyzed.
    feature_list : list of str
        List of column names in `dataframe` for which correlations
        will be computed.

    Returns
    -------
    altair.Chart
        An Altair bubble chart visualizing the correlation matrix.

    Raises
    ------
    TypeError
        If `dataframe` is not a pandas DataFrame.
    TypeError
        If `feature_list` is not a list.
    ValueError
        If any feature in `feature_list` does not exist in `dataframe`.

    """
    # check dataframe type
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame")
        
    # check feature_list type
    if not isinstance(feature_list, list):
        raise TypeError("feature_list must be a list")
        
    # check features exist in dataframe
    missing = set(feature_list) - set(dataframe.columns)
    if missing:
        raise ValueError(f"Features not found in dataframe: {missing}")
        
    # Compute correlation matrix and save as a dataframe
    corr_df = (
            dataframe[feature_list]
            .corr()
            .stack()
            .reset_index()
        )
    corr_df.columns = ['Feature 1', 'Feature 2', 'Correlation']
    corr_df['Absolute Correlation'] = corr_df['Correlation'].abs()
    
     
    bubble_chart = (
        alt.Chart(corr_df)
        .mark_circle()
        .encode(
            x=alt.X(
                'Feature 1:N',
                axis=alt.Axis(labelAngle=-45, labelFontSize=16, titleFontSize=20)
            ),
            y=alt.Y(
                'Feature 2:N',
                axis=alt.Axis(labelFontSize=16, titleFontSize=20)
            ),
            size=alt.Size(
                'Absolute Correlation:Q',
                scale=alt.Scale(domain=(0, 1)),
            legend=None ),
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
                alt.Tooltip('Correlation:Q', format='.3f')
            ]
        )
        .properties(
            width=580,    
            height=580,
            title=alt.TitleParams(
                "Correlation Bubble Chart",
                fontSize=30,
                fontWeight="normal",
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
    
    return bubble_chart


