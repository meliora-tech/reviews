from io import StringIO
from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from typing import Dict, Union

import pandas as pd
import requests




def download_nrc_lexicon():
    print("Downloading NRC Emotion Lexicon...")
    response = requests.get('https://raw.github.com/dinbav/LeXmo/master/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt')
    print("Completed downloading NRC Emotion Lexicon.")
    return response 

def load_nrc_lexicon(file: str = "") -> Union[None, pd.DataFrame]:
    """
    Load the NRC Emotion Lexicon from file

    :params file is the path to the file
    :type :str
    """

    try:
        if file == "":
            emolex_df = pd.read_csv("nrc_emotion_lexicon.txt", names = ["word", "emotion", "association"],
                                sep=r'\t', engine='python')
        else:
            emolex_df = pd.read_csv(file,
                                names=["word", "emotion", "association"],
                                sep=r'\t', engine='python')
    except Exception as e:
        print(e)
        emolex_df = None

    return emolex_df

def read_nrc_lexicon(file: str = "") -> pd.DataFrame:
    """
    Read the NRC file

    :params file is the path to the file
    :type :str
    """
    
    # Try and load the NRC Lexicon from file     
    emolex_df = load_nrc_lexicon(file)

    # Download, if possible the NRC Lexicon
    if emolex_df is None:
        response = download_nrc_lexicon()
        nrc       = StringIO(response.text)
        emolex_df = pd.read_csv(nrc,
                                names=["word", "emotion", "association"],
                                sep=r'\t', engine='python')

        # Save the data as a txt file 

    return emolex_df

def LeXmo(text: str, emolex_df: pd.DataFrame) -> Dict[str, float]:

    """"
      Takes text and adds if to a dictionary with 10 Keys  for each of the 10 emotions in the NRC Emotion Lexicon,
      each dictionay contains the value of the text in that emotions divided to the text word count
      INPUT: string
      OUTPUT: dictionary with the text and the value of 10 emotions

    Source: https://github.com/dinbav/LeXmo/blob/master/LeXmo/LeXmo.py

    :params text to run score on
    :type :str 

    :params emolex_df is the dataframe with the NRC Lexicon
    :type :pd.DataFrame

    :return: (Dict[str, str])
    """

        
    LeXmo_dict = {'text': text, 'anger': [], 'anticipation': [], 'disgust': [], 'fear': [], 'joy': [], 'negative': [],
                    'positive': [], 'sadness': [], 'surprise': [], 'trust': []}


    emolex_words = emolex_df.pivot(index='word',
                                   columns='emotion',
                                   values='association').reset_index()


    emolex_words.drop(emolex_words.index[0])

    emotions = emolex_words.columns.drop('word')

    stemmer = SnowballStemmer("english")

    document = word_tokenize(text)

    word_count = len(document)
    rows_list = []
    for word in document:
        word = stemmer.stem(word.lower())

        emo_score = (emolex_words[emolex_words.word == word])
        rows_list.append(emo_score)

    df = pd.concat(rows_list)
    df.reset_index(drop=True)

    for emotion in list(emotions):
        LeXmo_dict[emotion] = df[emotion].sum() / word_count

    return LeXmo_dict