import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns


def get_group_topics_prop(df_w_factions, groupby_col, topic_col='topic_words3', n_topics_per_group=50):
    """calculate topic proportions for each group value in group_col
     -INPUT:
        -df_w_factions: pd.DataFrame with factions and topics
        - group_col: str of colum name where grouping variable is
        - topic_col: str of column where topic names are
        - n_topics_per_group: int of number of max topics taken for each group. NB! might create 0 in groups
            where topic is with too low proportion but might preserve value for other group where proportion is higher
    -OUTPUT:
        -pd.DataFrame in long format with each group topic proportions"""
    df_fation_topics = pd.DataFrame(
        df_w_factions.groupby([groupby_col])[topic_col].value_counts(normalize=True)).stack().reset_index()
    df_fation_topics.columns = [groupby_col, topic_col, 'level_2', 'count']
    df_fation_topics = df_fation_topics.groupby(groupby_col).head(n_topics_per_group)
    return df_fation_topics


def create_group_vecs(df_topics, topic_col='topic_words3', group_col='fation', fillna=0):
    """create topic proportion vectors pd.DataFrame for each group value in group column
         -INPUT:
            - df_topics: pd.DataFrame with group and topic column, should come from get_group_topics_prop
            - group_col: str of colum name where grouping variable is
            - topic_col: str of column where topic names are
            - fillna: int, float with what missing data in tipic vectors is filled (if certain group doesn't
            have data in a specific topic
        -OUTPUT:
            -pd.DataFrame with topic proportion vectors for each group"""
    df_group_topic_vecs = df_topics[[group_col, topic_col, 'count']] \
        .pivot_table(values='count', index=topic_col, columns=group_col)
    if fillna is not None:
        df_group_topic_vecs = df_group_topic_vecs.fillna(fillna)
    return df_group_topic_vecs


def calc_group_topic_vec_cossim(df_group_topic_vecs, fill_diag_value=0):
    """calculate topic proportion vectors similarity using cosine similarity
    -INPUT:
        - df_group_topic_vecs: pd.DataFrame with topic prop vectors
        - fill_diag_value: int, float value which diagonal elements are to be filled in similarity heatmap
            (diagonal values are all 1, sometimes good to fill with 0 so that plot colors would have more variability
    -OUTPUT:
        -return topic proportions cosine similarity pd.DataFrame
         """
    df_topic_cossim = pd.DataFrame(cosine_similarity(df_group_topic_vecs.T))
    # diagonal elements are all 0, not adding any value, for visualization turn them 0
    np.fill_diagonal(df_topic_cossim.values, fill_diag_value)
    df_topic_cossim.columns = df_group_topic_vecs.columns
    df_topic_cossim.index = df_group_topic_vecs.columns
    return df_topic_cossim


def plot_heatmap(df, figsize=(7, 7)):
    """plot heatmap from df"""
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(df, cmap='Blues', robust=True, annot=True, ax=ax)


def create_plot_group_topvecs(df, n_topics, group_col='fation', topic_col='topic_words3', fillna=0,
                              figsize=(7, 7), return_data=True, fill_diag_value=0):
    """create topic proportion vector for each group value in group column and plot each group vector similarity heatmap
    (similarity measure is cosine similarity)
     -INPUT:
        - df: pd.DataFrame with group and topic column
        - group_col: str of colum name where grouping variable is
        - topic_col: str of column where topic names are
        - fillna: int, float with what missing data in tipic vectors is filled (if certain group doesn't
            have data in a specific topic
        - figsize: tuple of ints with plot dimensions
        - return_data: bool if data is returned
        - fill_diag_value: int, float value which diagonal elements are to be filled in similarity heatmap
            (diagonal values are all 1, sometimes good to fill with 0 so that plot colors would have more variability
    -OUTPUT:
        - groups topic proportions vectors similarity heatmap, if return_data, returns group topic proportions vectors
        and cosine similarity pd.DataFrame"""
    df_topics = get_group_topics_prop(df, group_col, topic_col, n_topics)
    df_group_topic_vecs = create_group_vecs(df_topics, topic_col=topic_col, group_col=group_col, fillna=fillna)
    df_topic_cossim = calc_group_topic_vec_cossim(df_group_topic_vecs, fill_diag_value)
    plot_heatmap(df_topic_cossim, figsize=figsize)
    if return_data:
        return df_group_topic_vecs, df_topic_cossim
