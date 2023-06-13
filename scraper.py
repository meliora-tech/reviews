"""
This file is the scraper for reviews on Hello Peter

Date  : 2023-06-05 
"""

import json
import numpy as np
import pandas as pd 
import random
import requests
import sys
import time 

from requests import Response 
from typing import Dict, List, Union

# Own modules
from topic_modeling import run_bertopic, run_lda, run_nmf, run_top2vec

# Constant URLS
BUSINESS_STATS_URL         = "https://api.hellopeter.com/api/consumer/business-stats/"
CATEGORIES_URL             = "https://api.hellopeter.com/api/categories"
HELLO_PETER_API_URL        = "https://api.hellopeter.com/api/"
LIST_OF_INDUSTRIES_URL     = "https://api.hellopeter.com/api/consumer/industries/list"
REVIEW_PAGE_URL            = "https://api.hellopeter.com/api/consumer/business/"
# INDUSTRY_URL        = f"https://api.hellopeter.com/api/consumer/industries/{industry_slug_name}/businesses"


# Constant PATHS
TRUST_INDEX_PATH       = "/reports/trust-index-over-time"
REVIEW_PATH            = "/reviews?page="



def parse_business_stats(data: Dict) -> pd.DataFrame:
    """
    Parse the contents from calling the Hello Peter Business Stats API 

    :params data is the dict from the output of the business stats api
    :type :Dict 

    :return: (pd.DataFrame)
    """

    keys = list(data.keys())

    # Overall stats 
    total_reviews   = data[keys[0]]
    review_average  = data[keys[1]]
    review_ratings  = data[keys[2]]
   
    review_stars   = [ r[0] for r in review_ratings['rows']]
    review_values  = [ r[1] for r in review_ratings['rows']]
    
    # Monthly stats
    monthly_stats  = data[keys[3]]
    try:
        monthly_trust_index       = [ m['trustIndex'] for m in monthly_stats['months']] 
        monthly_industry_ranking  = [ m['industryRanking'] for m in monthly_stats['months']] 
        monthly_month             = [ f"{m['month']}-{2023}" for m in monthly_stats['months']] 
    except:
        monthly_trust_index       = np.nan
        monthly_industry_ranking  = np.nan
        monthly_month             = [ f"{m['month']}-{2023}" for m in monthly_stats['months']]        

    # Create the results placeholder
    business_review_df                           = pd.DataFrame()

    # Store the monthly stats 
    try:
        business_review_df["month_trust_index"]      = monthly_month
        business_review_df["month_industry_ranking"] = monthly_industry_ranking
        business_review_df["month_trust_index "]     = monthly_trust_index
    except:
        business_review_df["month_trust_index"]      = monthly_month
        business_review_df["month_industry_ranking"] = np.nan
        business_review_df["month_trust_index "]     = np.nan    

    business_review_df["total_reviews"]          = total_reviews
    business_review_df["review_average"]         = review_average

    business_review_df["review_stars"]   = ""
    business_review_df["review_values"]  = np.nan
    temp_df = business_review_df

    for i in range(0,len(review_stars)):
        temp_df["review_stars"]           = review_stars[i]
        temp_df["review_values"]           = review_values[i]

        business_review_df = pd.concat([business_review_df, temp_df])

    return business_review_df


def get_business_stats(id: str) -> pd.DataFrame:
    """
    Get the business stats for the given company id from Hello Peter

    :params id is the Hello Peter id for the company
    :type :str 

    :return: (pd.DataFrame)
    """

    business_stats_url = f"{BUSINESS_STATS_URL}{id}"
    response           = make_request(business_stats_url)

    if response.status_code >= 200 and response.status_code < 300:
        content  = response.json()
        results  = parse_business_stats(content)

        return results 
    else:
        print(response)
        return pd.DataFrame()



def get_trust_index(id:str, start_date:str, end_date:str) -> pd.DataFrame:
    """
    Get the Trust Indices for the given company for the period between start_date and end_date

    :params id is the Hello Peter id for the company
    :type :str 

    :params start_date
    :type :str

    :params end_date
    :type :str 

    :return: (pd.DataFrame)
    """

    results            = pd.DataFrame()
    trust_index_params = f"?start_date={start_date}&end_date={end_date}"
    trust_index_url    = f"{HELLO_PETER_API_URL}{id}{TRUST_INDEX_PATH}{trust_index_params}" 

    response = make_request(trust_index_url)
    if response.status_code < 300 and response.status_code >= 200:
        content = response.json()
        rows = content['rows']    
        if len(rows) > 0:
            results = pd.DataFrame(rows) 
            results.rename(columns={0:'Date', 1:'Trust Index'}, inplace=True)
            results["ID"] = id 

    else:
        print(response)    
        
    return results


def make_request(url :str) -> Response:
    """
    Make a request to the Hello Peter url

    :params url is the Hello Peter API url
    :type :str

    :return: ()
    """

    response = requests.get(url)

    return response
     


def get_review_links(url) -> List[str]:
    """
    Collect all the review links for a given business

    :params url is the Hello Peter API url
    :type :str
    
    :return: (List[str])
    """
    response   = make_request(f"{url}1")
    content    = response.json()

    # Create all the page links
    total_pages     = content["last_page"]  
    pages           = range(1,total_pages+1)
    all_page_links  = [ f"{base_review_url}{p}" for p in pages]

    return all_page_links



def get_reviews(id: str) -> pd.DataFrame:
    """
    Get reviews for the given company based on its Hello Peter id
    
    :params id is the Hello Peter id for the company
    :type :str 

    :return: (pd.DataFrame)    
    """

    # Call the first page to get the total number of pages from the response
    base_review_url = f"{REVIEW_PAGE_URL}{id}{REVIEW_PATH}"

    # Get the review page links 
    all_page_links = get_review_links(base_review_url)

    # Placeholder for the review data
    all_reviews = pd.DataFrame()

    for link in all_page_links:
        try:
        
            response   = make_request(link)
            if response.status_code >=200 and response.status_code < 300:
                content = response.json()
                all_reviews = parse_reviews(content, all_reviews)
            else:
                print(response)
        except Exception as e:
            print(e)

        # Wait between 4 and 10 seconds
        rn = random.randint(4, 10)
        print(f"Waiting {rn} seconds before scraping again {id}...")
        time.sleep(rn)

    if len(all_reviews) > 0:
        all_reviews["ID"] = id 

    return all_reviews


def parse_reviews(content: Dict[str, Union[str, float, List[str]]], all_reviews: pd.DataFrame) -> pd.DataFrame:
    """
    Parse the review data fetched from Hello Peter

    :params content is the data from Hello Peter 
    :type :Dict[str, Union[str, float, List[str]]]

    :params all_reviews is a dataframe to save the results in 
    :type :pd.DataFrame

    :return: (pd.DataFrame)
    """
    review_data = content["data"]
    if len(review_data) > 0:
        for d in review_data:
            d_ = pd.DataFrame.from_dict(d, orient='index').T
            all_reviews = pd.concat([all_reviews, d_])

    return all_reviews

def main():
    """
    Main function to scrape Hello Peter
    """
    platforms_df = pd.read_csv("source.csv")

    collected_reviews = pd.DataFrame()
    for i in range(len(platforms_df)):
       
        platform_name  = platforms_df["name"][i]
        hello_peter_id = platforms_df["id"][i]

        reviews = get_reviews(hello_peter_id)
        collected_reviews = pd.concat([collected_reviews, reviews])

    collected_reviews.to_json("hello_peter_collected_reviews_20230606.json", orient="records" )


if __name__ == "__main__":
    main()
 

