"""
All the sentiment functions

Currently the following are implemented:
1. AFINN
2. NRC
3. VADER

Author: Vibin Data Team
Date: 2023-06-06
"""

import numpy as np
import pandas as pd 

from afinn import Afinn
# from LeXmo import LeXmo
from nltk import tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from progress.bar import Bar

# Own modules
from cleaning import clean_text
from lexmo import (read_nrc_lexicon, LeXmo)


def get_nrc_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the NRC affect scores for each text 

    :params df is a dataframe 
    :type :pd.DataFrame

    :return: (pd.DataFrame)
    """

    # Calculate the emotion scores and remove the text from the results
    nrc_lexicon_df = read_nrc_lexicon()
    emo        = [LeXmo(clean_text(review), nrc_lexicon_df)  for review in df['review_content']]
    clean_emo  = [e  for e in emo if e.pop('text', None) ]

    # Placeholder for NRC emotion scores
    nrc  = {"anger":[], "anticipation": [], "disgust": [], "fear": [], "joy": [], 
            "negative": [], "positive": [], "sadness": [], "surprise": [], "trust": [] }

    
    # Save the scores in the nrc dict
    bar = Bar("Saving scores...", max=len(clean_emo))
    for cemo in clean_emo:
        for k,v in cemo.items():
            values = nrc[k]
            values = values + [v]
            nrc[k] = values
        bar.next()
    bar.finish()

    # Convert NRC dict to Dataframe
    df_nrc = pd.DataFrame.from_dict(nrc, orient='index').T
    
    # Column bind the df_nrc and df 
    df = pd.concat([df, df_nrc], axis=1)

    return df 


def get_vader_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the VADER compound scores 

    :params df is the dataframe
    :type :pd.DataFrame

    :return: (pd.DataFrame)
    """

    review_compound_scores = []

    bar = Bar("Calculating VADER compound scores...", max=len(df))
    for i in range(len(df)):
        sentences             = tokenize.sent_tokenize(df['review_content'][i])
        sid                   = SentimentIntensityAnalyzer()
        compound_scores       = [ sid.polarity_scores(s)['compound']  for s in sentences]
        final_compound_score  = round(sum(compound_scores)/len(compound_scores),2)

        review_compound_scores.append(final_compound_score)
        bar.next()
    bar.finish()

    
    df["compound_scores"] = review_compound_scores 

    return df 



def get_afinn_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the AFINN scores 

    :params df 
    :type :pd.DataFrame 

    :return: (pd.DataFrame)
    """

    afinn = Afinn()

    # Calculate the sentiment scores
    sentiment_scores       = [ afinn.score(clean_text(review)) for review in df['review_content']]
    df['sentiment_scores'] = sentiment_scores
    
    # Assign the scores to a category
    sentiment_conditions = [(df['sentiment_scores'] > 0), (df['sentiment_scores'] == 0), (df['sentiment_scores'] < 0) ]
    sentiment_choices    = ["positive", "neutral", "negative"]
    sentiment_category   = np.select(sentiment_conditions, sentiment_choices, default="neutral") 
    df['sentiment_category'] = sentiment_category


    return df 