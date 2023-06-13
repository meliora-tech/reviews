"""
Run all the topic models 

Date: 2023-05-23
"""

import pandas as pd

from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from top2vec import Top2Vec
from typing import List, Tuple, Union

# Own modules
from plot_graphs import plot_wordcloud

def run_lda(corpus: List[str],  topics: int=7, max_df: float = 0.95, min_df: float = 2, random_state: int = 1) -> Tuple[LatentDirichletAllocation, CountVectorizer]:
    """"
    Run LDA 
    
    :params corpus is a list of strings 
    :type :List[str]
 
    :params topics is the number of topics to discover
    :type :int 

    :params max_df 
    :type :float 

    :params min_df 
    :type :float

    :params random_state 
    :type :int
    

    :return: (Tuple[LatentDirichletAllocation, CountVectorizer])
    """
    # Create the document term
    cv = CountVectorizer(max_df=max_df, min_df=min_df, stop_words='english')
    dtm = cv.fit_transform(corpus)

    LDA = LatentDirichletAllocation(n_components=topics,random_state=random_state)
    LDA.fit(dtm)


    return LDA, cv

def run_bertopic(corpus: List[str]):
    """
    Run BERTopic
    
    :params corpus
    :type :List[str]


    """

    topic_model = BERTopic()
    topics, probs = topic_model.fit_transform(corpus)

    return topic_model, topics, probs

def get_bertopic_topic(model: BERTopic, topic:int =0):
    """
    Get the BERTopic results
    
    :params model 
    :type :BERTopic

    :params topic is the number of the topic 
    :type :int
    """

    return model.get_topic(topic)


def get_bertopic_document_info(model: BERTopic, corpus: List[str]):
    """
    Get the document info 

    :params model 
    :type :BERTopic   
    
    :params corpus
    :type :List[str]

    """

    return model.get_document_info(corpus)



def run_lsa():
    """
    Run LSA 

    """
    pass

def run_nmf(corpus: List[str], topics:int = 2 ,max_df: float = 0.8, min_df: float = 0.01) -> Tuple[NMF, TfidfVectorizer]:
    """"
    Run NMF 

    See: https://towardsdatascience.com/nmf-a-visual-explainer-and-python-implementation-7ecdd73491f8

    :params corpus
    :type :List[str]

    :params topics is the number of topics to discover
    :type :int 

    :params max_df 
    :type :float 

    :params min_df 
    :type :float

    :return: (Tuple[NMF, TfidfVectorizer])
    """
    tf_idf  = TfidfVectorizer(ngram_range = (1,1), max_df = max_df, min_df = min_df)
    data_tv = tf_idf.fit_transform(corpus)

    dtm = pd.DataFrame(data_tv.toarray(), columns=tf_idf.get_feature_names_out())

    nmf_model = NMF(topics)

    doc_topic = nmf_model.fit_transform(dtm)

    return nmf_model, tf_idf


def run_top2vec(corpus: List[str], embedding_model: str = '', speed: str = 'fast-learn', min_count: int = 50) -> Top2Vec:
    """
    Run a top2vec on the given corpus 

    :params corpus
    :type :List[str]


    :params embedding_model is type of joint word and document embedding. 
    ['universal-sentence-encoder', 'universal-sentence-encoder-multilingual', 'distiluse-base-multilingual-cased']
    :type :str 

    :params speed is how fast to train the model
    ['fast-learn', 'learn', 'deep-learn']
    :type :str

    :params min_count is minium number of total word frequency to consider.
    :type :int

    :return: (Top2Vec)
    """

    if embedding_model == '':
        model = Top2Vec(corpus, speed=speed, min_count=min_count)
    else:
        model = Top2Vec(corpus, embedding_model=embedding_model, speed=speed, min_count=min_count)

    return model 



def get_top2vec_results(model: Top2Vec, **kwargs):
    """
    Get the results from a Top2Vec model 

    :params model is a Top2vec model 
    :type :Top2Vec 

    **kwargs 

        :param n_topics is the number of topics identified
        :type :bool

        :param topic_nums is the unique index of every topic will be returned
        :type :bool

        :param topic_sizes is the number of documents most similar to each topic.
        :type :bool 

        :param topic_words - for each topic the top 50 words are returned, in order of semantic similarity to topic.
        :type :bool

        :param word_scores - for each topic the cosine similarity scores of the top 50 words to the topic are returned.
        :type :bool 

    """

    if len(kwargs) != 0:
        pass 

    # Get number of topics
    n_topics = model.get_num_topics()

    # Get Topic Sizes
    topic_sizes, topic_nums = model.get_topic_sizes()

    # Get Topics
    topic_words, word_scores, topic_nums = model.get_topics(n_topics)


    return n_topics, topic_sizes, topic_words, word_scores, topic_nums

def display_topics(model: Union[LatentDirichletAllocation, NMF], feature_names: Union[TfidfVectorizer, CountVectorizer], num_top_words: int = 10):
    """"
    Given a topic model give the topics 

    :params model is the type of topic model
    :type :Union[LatentDirichletAllocation, NMF]

    :params feature_names is the feature names from Vectors
    :type :Union[TfidfVectorizer, CountVectorizer]

    :params num_top_words is the number of words per topic to extract
    :type :int 


    """

    for index,topic in enumerate(model.components_):
        print(f'topic #{index} : ')
        print([list(feature_names.vocabulary_.keys())[i] for i in topic.argsort()[-num_top_words:]])


def display_wordclouds(model         : Union[LatentDirichletAllocation, NMF], 
                      feature_names  : Union[TfidfVectorizer, CountVectorizer], 
                      num_top_words  : int  = 50, 
                      show_wordcloud : bool = False, 
                      use_freq       : bool = False):
    """
    Given a topic model give the topics 

    :params model is the type of topic model
    :type :Union[LatentDirichletAllocation, NMF]

    :params feature_names is the feature names from Vectors
    :type :Union[TfidfVectorizer, CountVectorizer]

    :params num_top_words is the number of words per topic to extract
    :type :int 

    :params show_wordcloud is show the word cloud or save them
    :type :bool

    :params use_freq is the choice to use the frequencies to display the size of the word.
    :type :bool

    :return: None
    """

    for index,topic in enumerate(model.components_):
        print(f'Generating wordcloud for topic #{index} ')

        words_list = [list(feature_names.vocabulary_.keys())[i] for i in topic.argsort()[-num_top_words:]]

        words = ''
        words += " ".join(words_list) + " "

        plot_wordcloud(words, show=False, name=f"topic-{index}")