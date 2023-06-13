import pandas as pd
import string 

from typing import Dict, List 

def create_id_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    This is to create fake names for the companies

    :params df is the dataframe 
    :type :pd.DataFrame 

    :return: (pd.DataFrame)
    """

    pseudo_names   = string.ascii_uppercase
    business_names = df["business_name"].unique()
    df["id_names"] = df["business_name"]
    for i,bn in enumerate(business_names):
           df.loc[df["business_name"] == bn,"business_name"] = pseudo_names[i]

    return df  


    
def black_color_func(word, font_size, position,orientation,random_state=None, **kwargs):
    return("hsl(0,100%, 1%)")