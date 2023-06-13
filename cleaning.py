
"""
All the text cleaning functions

Date: 2023-06-06
"""



import re 


def clean_text(text: str) -> str:
    """
    Clean the given sentence

    :params text that has to be cleaned 
    :type :str 

    :return: (str)
    """

    # remove html tags 
    clean = re.compile('<.*?>')
    text  = re.sub(clean, '', text)

    return text 
