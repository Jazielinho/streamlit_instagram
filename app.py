
import pandas as pd
import os
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
from sklearn.preprocessing import normalize
import numpy as np
from streamlit_plotly_events import plotly_events



# directory = '/media/jahaziel/Datos/proyectos/Smarketing/dataset/only_instagram/2023-06-03'

# directorio actual automático
directory = __file__.split('app.py')[0]
# print(directory)

WIDTH = 1000
HEIGHT = 600

min_topic_sizes = list(set([x.split('.csv')[0].split('_')[-1] for x in os.listdir(directory) if x.endswith('.csv')]))
min_topic_sizes = sorted([int(x) for x in min_topic_sizes])
min_topic_sizes = [str(x) for x in min_topic_sizes]

st.set_page_config(page_title='Prueba Instagram', page_icon=':chess_pawn:', layout='wide')

st.title(f'''Prueba Instagram''')

min_topic_size = st.radio('Mínimo tamaño de Topic', tuple(min_topic_sizes))

topics_over_time_df = pd.read_csv(f'''{directory}/topics_over_time_{min_topic_size}.csv''')
hierarchical_topics_df = pd.read_csv(f'''{directory}/hierarchical_topics_{min_topic_size}.csv''')
best_text_by_topics_df = pd.read_csv(f'''{directory}/best_text_by_topics_{min_topic_size}.csv''')
topic_word_weights_df = pd.read_csv(f'''{directory}/topic_word_weights_{min_topic_size}.csv''')
topic_labels_df = pd.read_csv(f'''{directory}/topic_labels_{min_topic_size}.csv''')
info_hashtags_df = pd.read_csv(f'''{directory}/info_hashtags_{min_topic_size}.csv''')
info_emojis_df = pd.read_csv(f'''{directory}/info_emojis_{min_topic_size}.csv''')
text_info_df = pd.read_csv(f'''{directory}/text_info_{min_topic_size}.csv''', lineterminator='\n')

text_info_df['date'] = pd.to_datetime(text_info_df['date']).dt.date
topics_over_time_df['Timestamp'] = pd.to_datetime(topics_over_time_df['Timestamp']).dt.date

for var in ['retweet_count', 'reply_count', 'like_count', 'quote_count', 'impression_count']:
    if var in text_info_df.columns:
        text_info_df[var] = text_info_df[var].fillna(0).astype(int)
    else:
        text_info_df[var] = 0

# text_info_df = text_info_df[text_info_df['date'] >= datetime.date(2023, 4, 1)]
# topics_over_time_df = topics_over_time_df[topics_over_time_df['Timestamp'] >= datetime.date(2023, 4, 1)]

dict_topic_labels = dict(zip(topic_labels_df['Topic'], topic_labels_df['Name']))

hierarchical_topics_df['Topics'] = hierarchical_topics_df['Topics'].apply(lambda x: eval(x))


# text_info_df = text_info_df[text_info_df['image_url'].apply(lambda x: str(x).lower() not in ['nan', 'none', ''])]

if 'image_url' not in text_info_df.columns:
    text_info_df['image_url'] = ''

text_images_info = text_info_df[text_info_df['image_url'].apply(lambda x: str(x).lower() not in ['nan', 'none', ''])].reset_index(drop=True)


if 'h_plot' not in st.session_state:
    st.session_state['h_plot'] = None

topics_over_time_df = topics_over_time_df[topics_over_time_df['Topic'] != -1]


if len(text_info_df) > 10000:
    text_info_df = text_info_df.sample(10000, random_state=123).reset_index(drop=True)

# ====================================================================================================
st.markdown(f'''## Estadísticas generales''')
st.markdown(f'''
* Número de textos publicados: {text_info_df.shape[0]}
* Periodo de tiempo: {text_info_df['date'].min()} - {text_info_df['date'].max()}
* Número de hashtags encontrados: {len(set(info_hashtags_df['hashtag']))}
''')

# ====================================================================================================

tab1, tab2, tab4, tab6 = st.tabs(['GENERAL', 'TOPICOS', 'HASHTAGS', 'IMAGES'])

with tab1:
    st.markdown(f'''#### Estadísticas''')
    general_statistics_df = text_info_df[['retweet_count', 'reply_count', 'like_count', 'quote_count', 'impression_count']].sum().astype(int).reset_index()
    general_statistics_df.columns = ['Estadística', 'Valor']
    st.table(general_statistics_df)

    st.markdown(f'''#### Estadísticas a lo largo del tiempo''')
    statistics_date_df = text_info_df.groupby('date')[['id', 'retweet_count', 'reply_count', 'like_count', 'quote_count', 'impression_count']].agg({'id': 'count', 'retweet_count': 'sum', 'reply_count': 'sum', 'like_count': 'sum', 'quote_count': 'sum', 'impression_count': 'sum'}).reset_index()
    statistics_date_df['engagement'] = statistics_date_df['retweet_count'] + statistics_date_df['reply_count'] + statistics_date_df['like_count'] + statistics_date_df['quote_count']
    statistics_date_df['engagement_rate'] = statistics_date_df['engagement'] / statistics_date_df['impression_count']

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=statistics_date_df['date'], y=statistics_date_df['id'], name='Número de textos publicados', mode='lines+markers'), secondary_y=False)
    fig.add_trace(go.Scatter(x=statistics_date_df['date'], y=statistics_date_df['engagement_rate'], name='Engagement rate', mode='lines+markers'), secondary_y=True)
    fig.update_layout(title_text='Número de textos publicados y engagement rate a lo largo del tiempo', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_layout(width=WIDTH, height=HEIGHT)
    st.plotly_chart(fig, use_container_width=True)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=statistics_date_df['date'], y=statistics_date_df['impression_count'], name='Número de impresiones', mode='lines+markers'), secondary_y=True)
    fig.add_trace(go.Bar(x=statistics_date_df['date'], y=statistics_date_df['retweet_count'], name='Número de retweets'), secondary_y=False)
    fig.add_trace(go.Bar(x=statistics_date_df['date'], y=statistics_date_df['reply_count'], name='Número de respuestas'), secondary_y=False)
    fig.add_trace(go.Bar(x=statistics_date_df['date'], y=statistics_date_df['like_count'], name='Número de likes'), secondary_y=False)
    fig.add_trace(go.Bar(x=statistics_date_df['date'], y=statistics_date_df['quote_count'], name='Número de quotes'), secondary_y=False)
    fig.update_layout(barmode='stack', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), title_text='Estadísticas a lo largo del tiempo')
    fig.update_layout(width=WIDTH, height=HEIGHT)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f'''#### Textos más populares''')
    columna_ordenar = st.selectbox('Elegir columna de ordenamiento', ['impression_count', 'retweet_count', 'reply_count', 'like_count', 'quote_count'])
    popular_texts_df = text_info_df.sort_values(columna_ordenar, ascending=False).head(10).reset_index(drop=True)
    popular_texts_df = popular_texts_df[['date', 'content', 'impression_count', 'retweet_count', 'reply_count', 'like_count', 'quote_count']]
    popular_texts_df.columns = ['Fecha', 'Texto', 'Impresiones', 'Retweets', 'Respuestas', 'Likes', 'Quotes']
    st.table(popular_texts_df)

    st.markdown(f'''#### Sentimiento de los textos ''')
    general_sentiment_df = text_info_df[['negative_score', 'neutral_score', 'positive_score']].mean().reset_index()
    general_sentiment_df.columns = ['Sentimiento', 'Valor']
    st.table(general_sentiment_df)

    sentiment_date_df = text_info_df.groupby('date')[['negative_score', 'neutral_score', 'positive_score']].mean().reset_index()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=sentiment_date_df['date'], y=sentiment_date_df['negative_score'], name='Sentimiento negativo'), secondary_y=False)
    fig.add_trace(go.Bar(x=sentiment_date_df['date'], y=sentiment_date_df['neutral_score'], name='Sentimiento neutral'), secondary_y=False)
    fig.add_trace(go.Bar(x=sentiment_date_df['date'], y=sentiment_date_df['positive_score'], name='Sentimiento positivo'), secondary_y=False)
    fig.update_layout(barmode='stack', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), title_text='Sentimiento de los textos a lo largo del tiempo')
    fig.update_layout(width=WIDTH, height=HEIGHT)
    st.plotly_chart(fig, use_container_width=True)


with tab2:
    if st.session_state['h_plot'] is None:
        df = pd.DataFrame({"topic": text_info_df['topic'].to_list()})
        df["doc"] = text_info_df['content'].to_list()
        df["topic"] = text_info_df['topic'].to_list()
        df["x"] = text_info_df['x'].to_list()
        df["y"] = text_info_df['y'].to_list()
        # df = df.sample(frac=0.3).reset_index(drop=True)
        distances = hierarchical_topics_df.Distance.to_list()
        max_distances = [distances[indices[-1]] for indices in np.array_split(range(len(hierarchical_topics_df)), 10)][::-1]
        for index, max_distance in enumerate(max_distances):
            mapping = {topic: topic for topic in df.topic.unique()}
            selection = hierarchical_topics_df.loc[hierarchical_topics_df.Distance <= max_distance, :]
            selection.Parent_ID = selection.Parent_ID.astype(int)
            selection = selection.sort_values("Parent_ID")
            for row in selection.iterrows():
                for topic in row[1].Topics:
                    mapping[topic] = row[1].Parent_ID
            mappings = [True for _ in mapping]
            while any(mappings):
                for i, (key, value) in enumerate(mapping.items()):
                    if value in mapping.keys() and key != value:
                        mapping[key] = mapping[value]
                    else:
                        mappings[i] = False
            df[f"level_{index+1}"] = df.topic.map(mapping)
            df[f"level_{index+1}"] = df[f"level_{index+1}"].astype(int)
        trace_names = []
        topic_names = {}
        for topic in range(hierarchical_topics_df.Parent_ID.astype(int).max()):
            if topic < hierarchical_topics_df.Parent_ID.astype(int).min():
                trace_name = dict_topic_labels[topic]
                topic_names[topic] = {"trace_name": trace_name[:40], "plot_text": trace_name[:40]}
                trace_names.append(trace_name)
            else:
                trace_name = f"{topic}_" + hierarchical_topics_df.loc[hierarchical_topics_df.Parent_ID == topic, "Parent_Name"].values[0]
                plot_text = "_".join([name[:20] for name in trace_name.split("_")[:3]])
                topic_names[topic] = {"trace_name": trace_name[:40], "plot_text": plot_text[:40]}
                trace_names.append(trace_name)
        all_traces = []
        for level in range(len(max_distances)):
            traces = []
            traces.append(
                    go.Scattergl(
                        x=df.loc[(df[f"level_{level+1}"] == -1), "x"],
                        y=df.loc[df[f"level_{level+1}"] == -1, "y"],
                        mode='markers+text',
                        name="other",
                        hoverinfo="text",
                        hovertext=df.loc[(df[f"level_{level+1}"] == -1), "doc"],# if not hide_document_hover else None,
                        showlegend=False,
                        marker=dict(color='#CFD8DC', size=5, opacity=0.5)
                    )
                )
            unique_topics = sorted([int(topic) for topic in df[f"level_{level+1}"].unique()])
            for topic in unique_topics:
                if topic != -1:
                    selection = df.loc[df[f"level_{level+1}"] == topic, :]
                    # if not hide_annotations:
                    selection.loc[len(selection), :] = None
                    selection["text"] = ""
                    selection.loc[len(selection) - 1, "x"] = selection.x.mean()
                    selection.loc[len(selection) - 1, "y"] = selection.y.mean()
                    selection.loc[len(selection) - 1, "text"] = topic_names[int(topic)]["plot_text"]

                    traces.append(
                        go.Scattergl(
                            x=selection.x,
                            y=selection.y,
                            text=None, #selection.text, # if not hide_annotations else None,
                            hovertext=selection.doc, # if not hide_document_hover else None,
                            hoverinfo="text",
                            name=topic_names[int(topic)]["trace_name"],
                            mode='markers+text',
                            marker=dict(size=5, opacity=0.5)
                        )
                    )
            all_traces.append(traces)
        nr_traces_per_set = [len(traces) for traces in all_traces]
        trace_indices = [(0, nr_traces_per_set[0])]
        for index, nr_traces in enumerate(nr_traces_per_set[1:]):
            start = trace_indices[index][1]
            end = nr_traces + start
            trace_indices.append((start, end))
        fig = go.Figure()
        for traces in all_traces:
            for trace in traces:
                fig.add_trace(trace)
        for index in range(len(fig.data)):
            if index >= nr_traces_per_set[0]:
                fig.data[index].visible = False
        fig = go.Figure()
        for traces in all_traces:
            for trace in traces:
                fig.add_trace(trace)

        for index in range(len(fig.data)):
            if index >= nr_traces_per_set[0]:
                fig.data[index].visible = False

        # Create and add slider
        steps = []
        for index, indices in enumerate(trace_indices):
            step = dict(
                method="update",
                label=str(index),
                args=[{"visible": [False] * len(fig.data)}]
            )
            for index in range(indices[1] - indices[0]):
                step["args"][0]["visible"][index + indices[0]] = True
            steps.append(step)

        sliders = [dict(
            currentvalue={"prefix": "Level: "},
            pad={"t": 20},
            steps=steps
        )]

        # Add grid in a 'plus' shape
        x_range = (df.x.min() - abs((df.x.min()) * .15), df.x.max() + abs((df.x.max()) * .15))
        y_range = (df.y.min() - abs((df.y.min()) * .15), df.y.max() + abs((df.y.max()) * .15))
        fig.add_shape(type="line",
                      x0=sum(x_range) / 2, y0=y_range[0], x1=sum(x_range) / 2, y1=y_range[1],
                      line=dict(color="#CFD8DC", width=2))
        fig.add_shape(type="line",
                      x0=x_range[0], y0=sum(y_range) / 2, x1=x_range[1], y1=sum(y_range) / 2,
                      line=dict(color="#9E9E9E", width=2))
        fig.add_annotation(x=x_range[0], y=sum(y_range) / 2, text="D1", showarrow=False, yshift=10)
        fig.add_annotation(y=y_range[1], x=sum(x_range) / 2, text="D2", showarrow=False, xshift=10)

        # Stylize layout
        fig.update_layout(
            sliders=sliders,
            template="simple_white",
            title={
                'text': f"<b>",
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(
                    size=22,
                    color="Black")
            },
            width=WIDTH,
            height=HEIGHT * 1.5,
        )

        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)

        st.session_state['h_plot'] = fig

    else:
        fig = st.session_state['h_plot']

    st.plotly_chart(fig, use_container_width=True)


    topic_count_sentiment_df = text_info_df.groupby('topic').agg({'id': 'count', 'negative_score': 'mean', 'neutral_score': 'mean', 'positive_score': 'mean'}).reset_index()
    topic_count_sentiment_df = topic_count_sentiment_df[topic_count_sentiment_df['topic'] != -1].sort_values('topic')
    topic_count_sentiment_df['topic_label'] = [dict_topic_labels[x] for x in topic_count_sentiment_df['topic']]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=topic_count_sentiment_df['topic_label'], y=topic_count_sentiment_df['id'], name='Cantidad de textos', mode='lines+markers'), secondary_y=True)
    fig.add_trace(go.Bar(x=topic_count_sentiment_df['topic_label'], y=topic_count_sentiment_df['negative_score'], name='Sentimiento negativo'), secondary_y=False)
    fig.add_trace(go.Bar(x=topic_count_sentiment_df['topic_label'], y=topic_count_sentiment_df['neutral_score'], name='Sentimiento neutral'), secondary_y=False)
    fig.add_trace(go.Bar(x=topic_count_sentiment_df['topic_label'], y=topic_count_sentiment_df['positive_score'], name='Sentimiento positivo'), secondary_y=False)
    fig.update_layout(barmode='stack', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), title_text='Cantidad de textos y sentimientos por tópico')
    fig.update_layout(width=WIDTH, height=HEIGHT * 1.5)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f'''#### Estadísticas por tópicos''')
    topic_statistics_df = text_info_df.groupby('topic').agg({'impression_count': 'sum', 'retweet_count': 'sum', 'reply_count': 'sum', 'like_count': 'sum', 'quote_count': 'sum'}).reset_index()
    topic_statistics_df = topic_statistics_df[topic_statistics_df['topic'] != -1].sort_values('topic')
    topic_statistics_df['topic_label'] = [dict_topic_labels[x] for x in topic_statistics_df['topic']]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=topic_statistics_df['topic_label'], y=topic_statistics_df['impression_count'], name='Número de impresiones', mode='lines+markers'), secondary_y=True)
    fig.add_trace(go.Bar(x=topic_statistics_df['topic_label'], y=topic_statistics_df['retweet_count'], name='Número de retweets'), secondary_y=False)
    fig.add_trace(go.Bar(x=topic_statistics_df['topic_label'], y=topic_statistics_df['reply_count'], name='Número de respuestas'), secondary_y=False)
    fig.add_trace(go.Bar(x=topic_statistics_df['topic_label'], y=topic_statistics_df['like_count'], name='Número de likes'), secondary_y=False)
    fig.add_trace(go.Bar(x=topic_statistics_df['topic_label'], y=topic_statistics_df['quote_count'], name='Número de quotes'), secondary_y=False)
    fig.update_layout(barmode='stack', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_layout(width=WIDTH, height=HEIGHT * 1.5)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f'''#### Evolución de tópicos ''')
    normalize_frequency = st.radio('Normalizar', ['No', 'Sí']) == 'Sí'
    colors = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#D55E00", "#0072B2", "#CC79A7"]
    topics_over_time_df = topics_over_time_df.sort_values(["Topic", "Timestamp"])
    topics_over_time_df['Timestamp'] = pd.to_datetime(topics_over_time_df['Timestamp'])
    topics_over_time_df["Name"] = topics_over_time_df.Topic.apply(lambda x: dict_topic_labels[x])
    fig = go.Figure()
    for index, topic in enumerate(topics_over_time_df.Topic.unique()):
        trace_data = topics_over_time_df.loc[topics_over_time_df.Topic == topic, :]
        topic_name = trace_data.Name.values[0]
        words = trace_data.Words.values
        if normalize_frequency:
            y = normalize(trace_data.Frequency.values.reshape(1, -1))[0]
        else:
            y = trace_data.Frequency
        fig.add_trace(go.Scatter(x=trace_data.Timestamp, y=y, mode='lines+markers', marker_color=colors[index % 7], hoverinfo="text", name=topic_name, hovertext=[f'<b>Topic {topic}</b><br>Words: {word}' for word in words]))
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    fig.update_layout( yaxis_title="Normalized Frequency" if normalize_frequency else "Frequency", template="simple_white", hoverlabel=dict(bgcolor="white", font_size=16, font_family="Rockwell"))
    fig.update_layout(width=WIDTH * 2, height=HEIGHT)
    st.plotly_chart(fig)

    st.markdown(f'''### Información por tópico ''')
    topicos_elegidos = st.multiselect('Elige los tópicos', [x for x in list(dict_topic_labels.keys()) if x != -1], [0, 1])
    if st.button('Ejecutar'):
        for topico_elegido in topicos_elegidos:
            st.markdown(f'''#### Tópico {topico_elegido}''')
            st.markdown(f'''##### Posibles nombres:''')
            st.markdown(f'''###### {dict_topic_labels[topico_elegido]}''')
            st.markdown(f'''##### Palabras más frecuentes''')
            best_words = topic_word_weights_df[topic_word_weights_df['topic'] == topico_elegido].sort_values('weight', ascending=False)
            fig = go.Figure()
            fig.add_trace(go.Bar(x=best_words['word'], y=best_words['weight'], name='Peso'))
            fig.update_layout(xaxis_title="Palabras", yaxis_title="Frecuencia")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(f'''##### Textos más representativos''')
            best_texts_2 = best_text_by_topics_df[best_text_by_topics_df['topic'] == topico_elegido]['text'].reset_index(drop=True)
            # best_texts_1 = best_text_by_topics_tfidf_df[best_text_by_topics_tfidf_df['labels'] == topico_elegido]['content'].reset_index(drop=True)
            best_texts = best_texts_2 #pd.concat([best_texts_1, best_texts_2], axis=0).reset_index(drop=True)
            st.write(best_texts)

            topic_text_df = text_info_df[text_info_df['topic'] == topico_elegido]

            st.markdown(f'''##### Estadísticas''')
            general_statistics_df = topic_text_df[['retweet_count', 'reply_count', 'like_count', 'quote_count', 'impression_count']].sum().astype(int).reset_index()
            general_statistics_df.columns = ['Estadística', 'Valor']
            st.table(general_statistics_df)

            statistics_date_df = topic_text_df.groupby('date')[['id', 'retweet_count', 'reply_count', 'like_count', 'quote_count', 'impression_count']].agg({'id': 'count', 'retweet_count': 'sum', 'reply_count': 'sum', 'like_count': 'sum', 'quote_count': 'sum','impression_count': 'sum'}).reset_index()
            statistics_date_df['engagement'] = statistics_date_df['retweet_count'] + statistics_date_df['reply_count'] + statistics_date_df['like_count'] + statistics_date_df['quote_count']
            statistics_date_df['engagement_rate'] = statistics_date_df['engagement'] / statistics_date_df['impression_count']

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=statistics_date_df['date'], y=statistics_date_df['id'], name='Número de textos publicados', mode='lines+markers'), secondary_y=False)
            fig.add_trace(go.Scatter(x=statistics_date_df['date'], y=statistics_date_df['engagement_rate'], name='Engagement rate', mode='lines+markers'), secondary_y=True)
            fig.update_layout(title_text='Número de textos publicados y engagement rate a lo largo del tiempo', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            fig.update_layout(width=WIDTH, height=HEIGHT)
            st.plotly_chart(fig, use_container_width=True)

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=statistics_date_df['date'], y=statistics_date_df['impression_count'], name='Número de impresiones', mode='lines+markers'), secondary_y=True)
            fig.add_trace(go.Bar(x=statistics_date_df['date'], y=statistics_date_df['retweet_count'], name='Número de retweets'), secondary_y=False)
            fig.add_trace(go.Bar(x=statistics_date_df['date'], y=statistics_date_df['reply_count'], name='Número de respuestas'), secondary_y=False)
            fig.add_trace(go.Bar(x=statistics_date_df['date'], y=statistics_date_df['like_count'], name='Número de likes'), secondary_y=False)
            fig.add_trace(go.Bar(x=statistics_date_df['date'], y=statistics_date_df['quote_count'], name='Número de quotes'), secondary_y=False)
            fig.update_layout(barmode='stack',legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), title_text='Estadísticas a lo largo del tiempo')
            fig.update_layout(width=WIDTH, height=HEIGHT)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(f'''##### Sentimiento de los textos ''')
            general_sentiment_df = topic_text_df[['negative_score', 'neutral_score', 'positive_score']].mean().reset_index()
            general_sentiment_df.columns = ['Sentimiento', 'Valor']
            st.table(general_sentiment_df)

            sentiment_date_df = topic_text_df.groupby('date')[['negative_score', 'neutral_score', 'positive_score']].mean().reset_index()
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(x=sentiment_date_df['date'], y=sentiment_date_df['negative_score'], name='Sentimiento negativo'), secondary_y=False)
            fig.add_trace(go.Bar(x=sentiment_date_df['date'], y=sentiment_date_df['neutral_score'], name='Sentimiento neutral'), secondary_y=False)
            fig.add_trace(go.Bar(x=sentiment_date_df['date'], y=sentiment_date_df['positive_score'], name='Sentimiento positivo'), secondary_y=False)
            fig.update_layout(barmode='stack', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), title_text='Sentimiento de los textos a lo largo del tiempo')
            fig.update_layout(width=WIDTH, height=HEIGHT)
            st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.markdown(f'''## HASHTAGS más usados ''')
    count_hashtag_df = info_hashtags_df['hashtag'].value_counts().reset_index().head(20)
    count_hashtag_df.columns = ['hashtag', 'count']
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=count_hashtag_df['hashtag'], y=count_hashtag_df['count'], name='Cantidad de hashtag', mode='lines+markers'), secondary_y=True)
    fig.update_layout(barmode='stack', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), title_text='Cantidad de hashtag')
    fig.update_layout(width=WIDTH, height=HEIGHT)
    st.plotly_chart(fig, use_container_width=True)

    hashtag_mencionadas = st.multiselect('Elige hashtags', count_hashtag_df['hashtag'], count_hashtag_df['hashtag'][:2])

    if st.button('Analizar hashtags'):
        for hashtag_elegida in hashtag_mencionadas:
            st.markdown(f'''#### hashtags: {hashtag_elegida}''')

            hashtag_text_df = text_info_df[text_info_df['id'].isin(info_hashtags_df[info_hashtags_df['hashtag'] == hashtag_elegida]['id'])]

            st.markdown(f'''##### Estadísticas''')
            general_statistics_df = hashtag_text_df[['retweet_count', 'reply_count', 'like_count', 'quote_count', 'impression_count']].sum().astype(int).reset_index()
            general_statistics_df.columns = ['Estadística', 'Valor']
            st.table(general_statistics_df)

            statistics_date_df = hashtag_text_df.groupby('date')[['id', 'retweet_count', 'reply_count', 'like_count', 'quote_count', 'impression_count']].agg({'id': 'count', 'retweet_count': 'sum', 'reply_count': 'sum', 'like_count': 'sum', 'quote_count': 'sum','impression_count': 'sum'}).reset_index()
            statistics_date_df['engagement'] = statistics_date_df['retweet_count'] + statistics_date_df['reply_count'] + statistics_date_df['like_count'] + statistics_date_df['quote_count']
            statistics_date_df['engagement_rate'] = statistics_date_df['engagement'] / statistics_date_df['impression_count']

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=statistics_date_df['date'], y=statistics_date_df['id'], name='Número de textos publicados', mode='lines+markers'), secondary_y=False)
            fig.add_trace(go.Scatter(x=statistics_date_df['date'], y=statistics_date_df['engagement_rate'], name='Engagement rate', mode='lines+markers'), secondary_y=True)
            fig.update_layout(title_text='Número de textos publicados y engagement rate a lo largo del tiempo', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            fig.update_layout(width=WIDTH, height=HEIGHT)
            st.plotly_chart(fig, use_container_width=True)

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=statistics_date_df['date'], y=statistics_date_df['impression_count'], name='Número de impresiones', mode='lines+markers'), secondary_y=True)
            fig.add_trace(go.Bar(x=statistics_date_df['date'], y=statistics_date_df['retweet_count'], name='Número de retweets'), secondary_y=False)
            fig.add_trace(go.Bar(x=statistics_date_df['date'], y=statistics_date_df['reply_count'], name='Número de respuestas'), secondary_y=False)
            fig.add_trace(go.Bar(x=statistics_date_df['date'], y=statistics_date_df['like_count'], name='Número de likes'), secondary_y=False)
            fig.add_trace(go.Bar(x=statistics_date_df['date'], y=statistics_date_df['quote_count'], name='Número de quotes'), secondary_y=False)
            fig.update_layout(barmode='stack', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), title_text='Estadísticas a lo largo del tiempo')
            fig.update_layout(width=WIDTH, height=HEIGHT)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(f'''##### Sentimiento de los textos ''')
            general_sentiment_df = hashtag_text_df[['negative_score', 'neutral_score', 'positive_score']].mean().reset_index()
            general_sentiment_df.columns = ['Sentimiento', 'Valor']
            st.table(general_sentiment_df)

            sentiment_date_df = hashtag_text_df.groupby('date')[['negative_score', 'neutral_score', 'positive_score']].mean().reset_index()
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(x=sentiment_date_df['date'], y=sentiment_date_df['negative_score'], name='Sentimiento negativo'), secondary_y=False)
            fig.add_trace(go.Bar(x=sentiment_date_df['date'], y=sentiment_date_df['neutral_score'], name='Sentimiento neutral'), secondary_y=False)
            fig.add_trace(go.Bar(x=sentiment_date_df['date'], y=sentiment_date_df['positive_score'], name='Sentimiento positivo'), secondary_y=False)
            fig.update_layout(barmode='stack', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), title_text='Sentimiento de los textos a lo largo del tiempo')
            fig.update_layout(width=WIDTH, height=HEIGHT)
            st.plotly_chart(fig, use_container_width=True)

with tab6:
    st.markdown(f'''#### Imágenes''')

    if len(text_images_info) >= 0:

        fig = go.Figure()
        _df = text_images_info[text_images_info['topic'] == -1]
        fig.add_trace(go.Scatter(x=_df['x'], y=_df['y'], hovertext=_df['content'], hoverinfo='text', mode='markers+text', name='Sin tópico', marker=dict(color='#CFD8DC', size=5, opacity=0.5), showlegend=False))
        all_topics = sorted(text_images_info['topic'].unique())
        for topic in all_topics:
            if int(topic) == -1:
                continue
            selection = text_images_info[text_images_info['topic'] == topic]
            label_name = dict_topic_labels[topic]
            fig.add_trace(go.Scatter(x=selection['x'], y=selection['y'], hovertext=selection['content'], hoverinfo='text', mode='markers+text', name=label_name, marker=dict(size=5, opacity=0.5)))
        x_range = [text_images_info['x'].min() - abs(text_images_info['x'].min() * 0.15), text_images_info['x'].max() + abs(text_images_info['x'].max() * 0.15)]
        y_range = [text_images_info['y'].min() - abs(text_images_info['y'].min() * 0.15), text_images_info['y'].max() + abs(text_images_info['y'].max() * 0.15)]
        fig.add_shape(type="rect", x0=sum(x_range) / 2, y0=y_range[0], x1=sum(x_range) / 2, y1=y_range[1], line=dict(color="#CFD8DC", width=2))
        fig.add_shape(type="rect", x0=x_range[0], y0=sum(y_range) / 2, x1=x_range[1], y1=sum(y_range) / 2, line=dict(color="#CFD8DC", width=2))
        fig.add_annotation(x=x_range[0], y=sum(y_range) / 2, text="D1", showarrow=False, yshift=10)
        fig.add_annotation(x=sum(x_range) / 2, y=y_range[1], text="D2", showarrow=False, xshift=10)
        fig.update_layout(template='simple_white', title={'text': "<b>", 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top', 'font': dict(size=22, color='Black')})
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.update_layout(width=WIDTH, height=HEIGHT * 1.5)

        selected_points = plotly_events(fig, click_event=True, hover_event=False, override_width=f'''200%''', override_height=HEIGHT * 1.5)

        for selected_point in selected_points:
            _x = selected_point['x']
            _y = selected_point['y']
            _filter_df = text_images_info[(text_images_info['x'] == _x) & (text_images_info['y'] == _y)]
            # print(_filter_df)
            _filter_info = _filter_df.iloc[0].to_dict()
            content = _filter_info['content']
            image_description = _filter_info['image_description']
            img_url = _filter_info['image_url']
            positive_score = _filter_info['positive_score']
            negative_score = _filter_info['negative_score']
            neutral_score = _filter_info['neutral_score']
            topico = _filter_info['topic']
            try:
                st.image(img_url, width=WIDTH // 2, caption=image_description)
                st.markdown(f'''##### Texto: \n{content} \n ##### Descripción de la imagen: \n{image_description}, \n ##### Sentimiento: \n positivo: {positive_score} - negativo: {negative_score} - neutral: {neutral_score}, \n ##### Tópico: {dict_topic_labels[topico]}''')
            except Exception as e:
                # print(e)
                pass
