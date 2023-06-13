"""
All the plot functions

Author: Vibin Data Team
Date: 2023-06-06
"""

import matplotlib.pyplot as plt
import pandas as pd 
import plotly.graph_objects as go 
import string 


from plotly.subplots import make_subplots
from typing import Union
from wordcloud import WordCloud

# Own modules
from utils import black_color_func

def plot_wordcloud(words: Union[str, pd.Series], show: bool =True,freq: bool=False, name: str="wordcloud"):
    """
    Plot a word cloud 

    :params words 
    :type  :Union[str, pd.Series]

    :params show the worcloud or save it as a png 
    :type :bool

    :params freq is whether to use the frequencies or actual words to plot 
    :type :bool

    :params name is the filename for the word cloud
    :type :str

    :return: None 
    """
    if freq:
      word_cloud  = WordCloud(width=3000, height=2000, background_color="white").generate_from_frequencies(words)
    else:
        word_cloud  = WordCloud(width=3000, height=2000, background_color="white").generate(words)
    
    word_cloud.recolor(color_func = black_color_func)

    plt.figure(figsize= [15,10])
    plt.imshow(word_cloud, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad = 0)
    
    if show:
        plt.show()
    else:
        plt.savefig(f"{name}.png")

def dot_plot_afinn_sentiment(df: pd.DataFrame):
    """
    Plot the Afinn sentiment

    :params df is a dataframe that has `sentiment_scores` and `business_name` columns
    :type :pd.DataFrame
    """

    


    fig = go.Figure()

    # Negative sentiment
    fig.add_trace(go.Scatter(
        x = df.loc[df["sentiment_scores"] < 0, "business_name"],
        y = df.loc[df["sentiment_scores"] < 0, "sentiment_scores"],
        name="Negative",
        mode="markers",
        marker=dict(
        color='crimson',
        line_color='crimson',
        size=12)
    ))

    # Positive sentiment
    fig.add_trace(go.Scatter(
        x = df.loc[df["sentiment_scores"] > 0, "business_name"],
        y = df.loc[df["sentiment_scores"] > 0, "sentiment_scores"],
        name="Positive",
        mode="markers",
        marker=dict(
        color='aquamarine',
        line_color='aquamarine',
        size=12)
    ))

    # Neutral sentiment
    fig.add_trace(go.Scatter(
        x = df.loc[df["sentiment_scores"] == 0, "business_name"],
        y = df.loc[df["sentiment_scores"] == 0, "sentiment_scores"],
        name="Neutral",
        mode="markers",
        marker=dict(
        color='rgba(156, 165, 196, 0.95)',
        line_color='rgba(156, 165, 196, 0.95)',
        size=12)
    ))  

    fig.update_layout(
        title="Afinn Sentiment for reviews of event ticketing companies in South Africa",
        xaxis=dict(
            showgrid=False,
            showline=True,
        ),
        paper_bgcolor='white',
        plot_bgcolor='white',
        hovermode='closest',
    )

    fig.show()


def box_plot_afinn_sentiment(df: pd.DataFrame):
    """
    Box plot of the Afinn sentiment

    :params df is a dataframe that has `sentiment_scores` and `business_name` columns
    :type :pd.DataFrame
    """
    fig = go.Figure()
    business_names = df["business_name"].unique()
    for bn in business_names:
        fig.add_trace(go.Box(y= df.loc[df["business_name"] == bn,"sentiment_scores"], name=bn))

    fig.update_layout(
        title="Afinn Sentiment for reviews of event ticketing companies in South Africa",
        xaxis=dict(
            showgrid=False,
            showline=True,
        ),
        paper_bgcolor='white',
        plot_bgcolor='white',
        hovermode='closest',
    )

    fig.show()


def dot_plot_nrc_sentiment():
    """
    
    """
    pass 

def box_plot_nrc_sentiment(df: pd.DataFrame):
    """
    Box plot of NRC sentiment
    
    :params df is a dataframe that has `positive`, `negative` and `business_name` columns
    :type :pd.DataFrame

    """

    fig = go.Figure()
    business_names = df["business_name"].unique()
    for bn in business_names:
        postive  = df.loc[df["business_name"] == bn , "positive"]
        negative = df.loc[df["business_name"] == bn, "negative"]
        pos_neg  = pd.concat([postive, negative]) 
       
        fig.add_trace(go.Box(y= pos_neg, name=bn))

    fig.update_layout(
        title="NRC Sentiment for reviews of event ticketing companies in South Africa",
        xaxis=dict(
            showgrid=False,
            showline=True,
        ),
        paper_bgcolor='white',
        plot_bgcolor='white',
        hovermode='closest',
    )

    fig.show()

def box_plot_nrc_emotions(df: pd.DataFrame):
    """
    
    """
    business_names = df["business_name"].unique()
    titles         = tuple(business_names) 
    total_cols     = 2 
    total_rows     = 5

    # Create the positions for each subplot
    pos = []
    for row in range(1,total_rows+1):
        for col in range(1,total_cols+1):
             pos.append([row, col])
    positions = {name:p   for name, p in zip(business_names, pos)}

    # Plot
    fig = make_subplots(rows=total_rows, cols=total_cols, subplot_titles=titles)
    

    business_names = df["business_name"].unique()
    affects        = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"] 

    for bn in business_names:
        df_subset = df[df["business_name"] == bn]
        for affect in affects:
            fig.add_trace(go.Box(y = df_subset[affect], name=affect), row=positions[bn][0], col=positions[bn][1])


    fig.update_layout(
        title="NRC Emotions for reviews of event ticketing companies in South Africa",
        xaxis=dict(
            showgrid=False,
            showline=True,
        ),
        paper_bgcolor='white',
        plot_bgcolor='white',
        hovermode='closest',
        showlegend=False
    )

    fig.show()


def box_plot_valence_sentiment(df: pd.DataFrame):
    """
    Box plot of VADER sentiment
    
    :params df is a dataframe that has `comppound_scores` and `business_name` columns
    :type :pd.DataFrame

    """
    
    fig = go.Figure()
    business_names = df["business_name"].unique()
    for bn in business_names:
        fig.add_trace(go.Box(y= df.loc[df["business_name"] == bn,"compound_scores"], name=bn))

    fig.update_layout(
        title="VADER Sentiment for reviews of event ticketing companies in South Africa",
        xaxis=dict(
            showgrid=False,
            showline=True,
        ),
        paper_bgcolor='white',
        plot_bgcolor='white',
        hovermode='closest',
    )

    fig.show()
