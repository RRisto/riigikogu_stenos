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


def create_group_vecs(df_topics, topic_col='topic_words3', group_col='fation', fillna=0, font_scale=1.):
    """create topic proportion vectors pd.DataFrame for each group value in group column
         -INPUT:
            - df_topics: pd.DataFrame with group and topic column, should come from get_group_topics_prop
            - group_col: str of colum name where grouping variable is
            - topic_col: str of column where topic names are
            - fillna: int, float with what missing data in tipic vectors is filled (if certain group doesn't
               have data in a specific topic
            - font_scale: float of font sizes in plot
        -OUTPUT:
            -pd.DataFrame with topic proportion vectors for each group"""
    sns.set(font_scale = font_scale)
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
                              figsize=(7, 7), return_data=True, fill_diag_value=0, font_scale=1.0):
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
        - font_scale: float of plot font scale
    -OUTPUT:
        - groups topic proportions vectors similarity heatmap, if return_data, returns group topic proportions vectors
        and cosine similarity pd.DataFrame"""
    
    df_topics = get_group_topics_prop(df, group_col, topic_col, n_topics)
    df_group_topic_vecs = create_group_vecs(df_topics, topic_col=topic_col, group_col=group_col, fillna=fillna, 
                                            font_scale=font_scale)
    df_topic_cossim = calc_group_topic_vec_cossim(df_group_topic_vecs, fill_diag_value)
    plot_heatmap(df_topic_cossim, figsize=figsize)
    if return_data:
        return df_group_topic_vecs, df_topic_cossim
    
    
def plot_catplot(df, x_col, y_col, group_col, x_axis_label, savefig, 
                 font_scale=1.2, sns_style='whitegrid', sharex=False,
                 sharey=False, color='lightblue', col_wrap=2,
                 height=2., aspect=2.5):
    """plot seaborn catplot: individual barplots for each level of group_col
    -INPUT:
        - df: pd.DataFrame containing data
        - x_col: str of column containing x-dimension data 
        - y_col: str of column containing y-dimension data
        - group_col: str of column containing  grouping variable
        - x_axis_label: str of x-axis label,
        - savefig: str of filename with path to save plot file
        - font_scale: float with scale of the plot font
        - sns_style: str of seaborn style
        - sharex: bool, if True all subplots have same scale on x-axis
        - sharey: bool, if True all subplots have same scale on y-axis
        - color: str of color of bars
        - col_wrap: int, in how many columns plots are organized
        - height: float, height of plot (from matplotlib)
        - aspect: float, aspect of plot (from matplotlib)
     -OUTPUT:
         plot with sub barplots (for each group_col level) and saved in savefig file"""
    sns.set(font_scale = font_scale)
    sns.set_style(sns_style)
    ax=sns.catplot(
        x=x_col,
        y=y_col,
        sharex=sharex,
        sharey=sharey,
        color=color,
        col=group_col,
        col_wrap=col_wrap, #Set the number of columns you want.
        data=df,
        kind='bar',
        height=height,
        aspect=aspect
    )
    ax.set_axis_labels(x_axis_label)
    plt.savefig(savefig, bbox_inches='tight')
    
def plot_topic_time_dynamics(df, colors, cluster_id=None, cluster_col='topic_cluster', time_col='year', topic_col='topic_words3',
                             figsize=(15,10), bbox_to_anchor=(1.09, 1.00), legend_fontsize=12,
                             x_tick_fontsize=12, y_tick_fontsize=12, x_axis_fontsize=12,
                             normalize_timesteps=False,
                             use_percentage=True):
    """function to plot topics time dynamics 
    -INPUT:
        -df: pd.DataFrame containing data, eahc row is some speaker text at some time
        -colors: list of colors used for plotting
        -cluster_id: if not None then int with topic cluster id used for plotting
        -cluster_col: str: name of the data cluster column
        -time_col: str of the data time column
        -topic_col: str of the data topic column
        -figsize: tuple of plot size
        -bbox_to_anchor: tuple of legend position
        -legend_fontsize: int of legend font size
        -x_tick_fontsize: int of x-axis tick labels font size
        -y_tick_fontsize: int of y-axis tick labels font size
        -x_axis_fontsize: int of x-axis title font size
        -normalize_timesteps: bool, if True normalizes data in each timestep (takes % of topics),
            useful for stacked are chart
        -use_percentage: bool, if not True counts each timestamp different topics and plots the scale of counts,
            otherwise plots the scale as %
    -OUTPUT:
        -plot of df topics (of cluster topics) dynamics"""
    fig, ax = plt.subplots(figsize=figsize)

    if not normalize_timesteps:
        df_topic_prop = df.groupby([time_col])[topic_col].\
            value_counts(normalize = use_percentage).\
            unstack()
        #keep only cluster topics for plotting
        if cluster_id is not None:
            df=df[df[cluster_col]==cluster_id]
        topic_cluster_names=df[topic_col].unique()
        columns2keep = topic_cluster_names
        df_topic_prop = df_topic_prop[columns2keep]
        df_topic_prop = df_topic_prop.reindex(sorted(df_topic_prop.columns), axis=1)
        ax = df_topic_prop.plot.area(ax=ax, color=colors)
    
    if normalize_timesteps:
        if cluster_id is not None:
            df=df[df[cluster_col]==cluster_id]
        df_plot=df.groupby([time_col])[topic_col].\
            value_counts(normalize = use_percentage).\
            unstack()
        df_plot = df_plot.reindex(sorted(df_plot.columns), axis=1)
        df_plot.plot.area(ax=ax, color=colors)
    
    ax.tick_params(axis='x', labelsize=x_tick_fontsize)
    ax.tick_params(axis='y', labelsize=y_tick_fontsize)
    ax.xaxis.get_label().set_fontsize(x_axis_fontsize)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], bbox_to_anchor=(1.0, 1.0), fontsize=legend_fontsize)

