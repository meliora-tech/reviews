"""
Analysis of the Hello Peter customer reviews for event ticketing companines in
South Africa

Date  : 2023-06-06
"""




import os 
import pandas as pd 
import plotly.graph_objects as go 
import time

import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from wordcloud import WordCloud
# Own modules
from plot_graphs import (box_plot_afinn_sentiment, box_plot_nrc_sentiment, box_plot_valence_sentiment, box_plot_nrc_emotions, plot_wordcloud)
from sentiment import (get_nrc_scores, get_vader_scores, get_afinn_scores)
from topic_modeling import (get_bertopic_topic, get_bertopic_document_info, get_top2vec_results, 
                            display_topics ,run_bertopic, run_top2vec, run_lda, display_wordclouds)
from utils import (create_id_names, black_color_func)


def main():

    # Read in the data 
    dir_     = os.path.join(os.path.abspath(''), "data", "scraped")
    filename = "XXXjson" 
    file     = os.path.join(dir_, filename)
    df       = pd.read_json(file)

    # Create anonymous company names
    df = create_id_names(df)
  
    # Calculate the AFINN scores
    df = get_afinn_scores(df)
   
    # Calculate the VADER compound scores
    df = get_vader_scores(df)

    # Calculate the NRC affect scores 
    start_nrc = time.time()
    df        = get_nrc_scores(df)
    end_nrc   = time.time()

    print(f"Time took for NRC scores: {end_nrc - start_nrc}")

    # Boxplot of the Afinn sentiment
    box_plot_afinn_sentiment(df)
    
    # Box plot the VADER Sentiment
    box_plot_valence_sentiment(df)

    # Box plot the nrc sentiment
    box_plot_nrc_sentiment(df)

    # Boxplot of the NRC emotions
    box_plot_nrc_emotions(df)

    #=======================
    # Topic Modeling section 
    #=======================

    # Remove stopwords
    stop_words = stopwords.words('english')
    df['review_content'] = df['review_content'].apply(lambda x: ' '.join([ word for word in x.split() if word not in (stop_words) ]))

    #=====================
    # Run Bertopic 
    #=====================

    start_bertopic = time.time()
    topic_model, topics, probs = run_bertopic(df['review_content'])
    end_bertopic  = time.time()
    print(f"Time took for Bertopic: {end_bertopic - start_bertopic}")
    bertopic_results = get_bertopic_document_info(topic_model, df['review_content'])
    print(bertopic_results)
    fig = topic_model.visualize_barchart()
    fig.write_html("bertopic.html")

    fig = topic_model.visualize_heatmap()
    fig.write_html("bertopic_heatmap.html")

    #===================
    # Run Top2Vec
    #==================
    start_top2vec = time.time()
    corpus_list   = list(df['review_content'])
    top2vec_model = run_top2vec(corpus_list)
    end_top2vec   = time.time()
    print(f"Time took for Top2Vec: {end_top2vec - start_top2vec}")
    n_topics, topic_sizes, topic_words, word_scores, topic_nums = get_top2vec_results(top2vec_model) 

    # Plot the word cloud 
    wordcloud_df = pd.DataFrame({"scores": word_scores[0]} , index=topic_words[0])
    plot_wordcloud(wordcloud_df["scores"], show=False, freq=True, name="topic2vec_wc")
 
    #============
    # Run LDA
    #============
    start_lda  = time.time()
    LDA, cv    = run_lda(df['review_content'], topics=5)
    end_lda    = time.time()
    print(f"Time took for LDA: {end_lda - start_lda}")

    display_wordclouds(LDA, cv)

if __name__ == "__main__":
    main()