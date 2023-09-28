import pandas as pd 
import streamlit as st 
from streamlit_echarts import st_echarts
from utils import load_data, create_sidebar_widgets
def main():

    # Loading data 
    data_path= 'data/players.parquet'
    df= load_data(data_path)

    # Extracing minimum and maximum age 
    min_age= df['Âge'].min()
    max_age= df['Âge'].max()

    # Position/roles templates
    templates = {
        "Striker": ['90s', 'Joueur', 'Équipe', 'Place', 'Âge', 'xG par 90', 'Tirs par 90', 'Touches de balle dans la surface de réparation sur 90',
                     'Actions défensives réussies par 90', 'Duels aériens gagnés par 90', 'Dribbles réussis par 90', 'xG/Tir', 
                     'Passes réceptionnées par 90'],

        "Winger/Attacking Midfielder": ['90s', 'Joueur', 'Équipe', 'Place', 'Âge', 'xG par 90', 'xA par 90', 'Tirs par 90', 
                                        'Touches de balle dans la surface de réparation sur 90', 'Сentres précises, %', 'Fautes subies par 90', 
                                        'Dribbles réussis par 90', 'Passes vers la surface de réparation précises, %', 'Interceptions PAdj'],

        "Midfielder": ['90s', 'Joueur', 'Équipe', 'Place', 'Âge', 'Passes précises, %', 'Passes progressives par 90', 'xA par 90',
                       'Dribbles réussis par 90', 'Fautes subies par 90', 'Interceptions PAdj', 'Courses progressives par 90'],

        "Defender": ['90s', 'Joueur', 'Équipe', 'Place', 'Âge', 'Passes réceptionnées par 90', 'Passes en avant précises, %', 'Interceptions PAdj', 
                     'Tacles glissés PAdj', 'Fautes par 90', 'Duels aériens par 90', 'Duels aériens gagnés, %'],

        "Full Wing Back": ['90s', 'Joueur', 'Équipe', 'Place', 'Âge', 'Passes réceptionnées par 90', 'Passes en avant précises, %',
                            'Interceptions PAdj', 'Tacles glissés PAdj', 'Fautes par 90', 'Duels aériens par 90', 'Duels aériens gagnés, %',
                              'Сentres précises, %', 'xA par 90']
    }

    st.titile('Player Similarity App')

    # Display & collect sidebar widgets values
    selected_player, selected_template, age_interval, nineties_interval, distance_metric, selected_positions, selected_leagues = create_sidebar_widgets(df)

    # Get and enable weightening of template's metrics
    template_values= templates[selected_template]
    metrics= template_values[5:]

    metrics_weights= {}
    st.sidebar.write('Custom Metric Weights:')
    for metric in metrics:
        weight = st.sidebar.slider(f'{metric}', 0.0, 1.0, 1.0, 0.1)
        metrics_weights[metric] = weight

    # Button to enerate similar players
    button= st.button('Similar players')

    # Process to generate similar players

    if button:
        similar_players_df= process_data(df, selected_player, template_values, metrics_weights, distance_metric)
