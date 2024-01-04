import matplotlib.pyplot as plt
import networkx as nx
import plotly.offline as py
import plotly.graph_objects as go

def plot_topic_pairs_ntw(topic_pairs_count, weight_col='pair_count', 
                         weight_div_fac=1000, title='Topic Co-occurrence Network',
                        ylabel_adjust_fac=0.03, figsize=(10, 10),
                        font_size=10, node_size_col='pair_count',
                        node_size_div_fac=1, alpha=0.7, name1_col='topic1',
                        name2_col='topic2'):
    G = nx.Graph()
    for _, row in topic_pairs_count.iterrows():
        G.add_edge(row[name1_col], row[name2_col], weight=row[weight_col])
        
    pos_kkl = nx.kamada_kawai_layout(G)
    f, ax = plt.subplots(figsize=figsize)

    d = dict(nx.degree(G))
    edges = G.edges()
    weights = [G[u][v]['weight']/weight_div_fac for u,v in edges]
    
    node_sizes = {}
    for u, v, w in topic_pairs_count[[name1_col, name2_col, node_size_col]].values:
        node_sizes[u] = w
        node_sizes[v] = w
    nx.set_node_attributes(G, node_sizes, name="size")

    nx.draw_networkx(G, pos_kkl, 
            with_labels=False, 
            node_size=[v/ node_size_div_fac for v in node_sizes.values()],
            nodelist=d.keys(),  
            width=weights, 
            edge_color='grey',
            alpha=alpha)
    
    pos_labels = {}
    for node, coords in pos_kkl.items():
        pos_labels[node] = (coords[0], coords[1] + ylabel_adjust_fac)

    nx.draw_networkx_labels(G, pos = pos_labels, font_size=font_size)

    # Set title
    ax.set_title(title, 
                 fontdict={'fontsize': 17,
                'fontweight': 'bold',
                'color': 'salmon', 
                'verticalalignment': 'baseline',
                'horizontalalignment': 'center'}, 
                 loc='center')
    # Set edge color
    plt.gca().collections[0].set_edgecolor("#000000")
    
 #basically same thing but with plotly
def plot_nx_(topic_pairs_count, weight_col='pair_count', 
                         weight_div_fac=1000, 
                        node_size_col='pair_count',
                         alpha=0.7,  name1_col='topic1',
                        name2_col='topic2'):
    G = nx.Graph()
    for _, row in topic_pairs_count.iterrows():
        G.add_edge(row[name1_col], row[name2_col], weight=row[weight_col])
        
    pos_kkl = nx.kamada_kawai_layout(G)
    edges = G.edges()
    
    node_sizes = {}
    for u, v, w in topic_pairs_count[[name1_col, name2_col, node_size_col]].values:
        node_sizes[u] = w
        node_sizes[v] = w
    nx.set_node_attributes(G, node_sizes, name="size")
    return G, pos_kkl


def make_edge(x, y, text, width):
    return  go.Scatter(x         = x,
                       y         = y,
                       line      = dict(width = width,
                                   color = 'cornflowerblue'),
                       hoverinfo = 'text',
                       text      = ([text]),
                       mode      = 'lines')

               
def plot_network_plotly(topic_pairs_count, weight_col='pair_count', 
                        weight_div_fac=1000,
                        node_size_col='pair_count',
                        alpha=0.7, name1_col='topic1',
                        name2_col='topic2', node_size_fac=50, savefile=None):
    
    g, pos_kkl=plot_nx_(topic_pairs_count, weight_col, weight_div_fac,
                         node_size_col, alpha, name1_col,
                        name2_col)
    
    #plotly part:
    edge_trace = []
    for edge in g.edges():

        if g.edges()[edge]['weight'] > 0:
            char_1 = edge[0]
            char_2 = edge[1]
            x0, y0 = pos_kkl[char_1]
            x1, y1 = pos_kkl[char_2]
            text   = char_1 + '--' + char_2 + ': ' + str(g.edges()[edge]['weight'])

            trace  = make_edge([x0, x1, None], [y0, y1, None], text, 
                               width = 20*g.edges()[edge]['weight']**2)
            edge_trace.append(trace)
            
    # Make a node trace
    node_trace = go.Scatter(x         = [],
                            y         = [],
                            text      = [],
                            textposition = "top center",
                            textfont_size = 10,
                            mode      = 'markers+text',
                            hoverinfo = 'text',
                            marker    = dict(color = [],
                                             size  = [],
                                             line  = None))
    # For each node in midsummer, get the position and size and add to the node_trace
    for node in g.nodes():
        x, y = pos_kkl[node]
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        node_trace['marker']['color'] += tuple(['cornflowerblue'])
        node_trace['marker']['size'] += tuple([node_size_fac*g.nodes()[node]['size']])
        node_trace['text'] += tuple(['<b>' + node + ' '+str(round(g.nodes()[node]['size'],3))+'</b>'])
        
    # Customize layout
    layout = go.Layout(
        paper_bgcolor='rgba(0,0,0,0)', # transparent background
        plot_bgcolor='rgba(0,0,0,0)', # transparent 2nd background
        xaxis =  {'showgrid': False, 'zeroline': False}, # no gridlines
        yaxis = {'showgrid': False, 'zeroline': False}, # no gridlines
        )
    # Create figure
    fig = go.Figure(layout = layout)
    # Add all edge traces
    for trace in edge_trace:
        fig.add_trace(trace)
    # Add node trace
    fig.add_trace(node_trace)
    # Remove legend
    fig.update_layout(showlegend = False)
    # Remove tick labels
    fig.update_xaxes(showticklabels = False)
    fig.update_yaxes(showticklabels = False)
    # Show figure
    fig.show()
    if savefile is not None:
        fig.write_html(savefile)