# Imports
import streamlit as st
from utils import load_data, process_data, filter_players, generate_radar_chart_option, convert_df, generate_pdf_report
from streamlit_echarts import st_echarts


def main():

    # Load data
    data_path = 'data/players.parquet'
    df = load_data(data_path)

    # Extract min and max age
    min_age = df['Âge'].min()
    max_age = df['Âge'].max()

    # Define templates and the metrics they inlcude
    templates_info = {
        "Striker": {
            "Metrics": {
                "90s": "Nombre de matchs joués sur 90 minutes",
                "xG par 90": "Expected Goals par 90 minutes",
                "Tirs par 90": "Nombre de tirs par 90 minutes",
                "Touches de balle dans la surface de réparation sur 90": "Nombre de touches de balle dans la surface de réparation par 90 minutes",
                "Actions défensives réussies par 90": "Nombre d'actions défensives réussies par 90 minutes",
                "Duels aériens gagnés par 90": "Nombre de duels aériens gagnés par 90 minutes",
                "Dribbles réussis par 90": "Nombre de dribbles réussis par 90 minutes",
                "xG/Tir": "Ratio xG/Tir",
                "Passes réceptionnées par 90": "Nombre de passes réceptionnées par 90 minutes",
            }
        },
        "Winger/Attacking Midfielder": {
            "Metrics": {

                "xG par 90": "Expected Goals par 90 minutes",
                "xA par 90": "Expected Assists par 90 minutes",
                "Tirs par 90": "Nombre de tirs par 90 minutes",
                "Touches de balle dans la surface de réparation sur 90": "Nombre de touches de balle dans la surface de réparation par 90 minutes",
                "Сentres précises, %": "Pourcentage de centres précis",
                "Fautes subies par 90": "Nombre de fautes subies par 90 minutes",
                "Dribbles réussis par 90": "Nombre de dribbles réussis par 90 minutes",
                "Passes vers la surface de réparation précises, %": "Pourcentage de passes vers la surface de réparation précises",
                "Interceptions PAdj": "Interceptions par 90 minutes (PAdj)",
            }
        },
        "Midfielder": {
            "Metrics": {
                "Passes précises, %": "Pourcentage de passes précises",
                "Passes progressives par 90": "Nombre de passes progressives par 90 minutes",
                "xA par 90": "Expected Assists par 90 minutes",
                "Dribbles réussis par 90": "Nombre de dribbles réussis par 90 minutes",
                "Fautes subies par 90": "Nombre de fautes subies par 90 minutes",
                "Interceptions PAdj": "Interceptions par 90 minutes (PAdj)",
                "Courses progressives par 90": "Nombre de courses progressives par 90 minutes",
            }
        },
        "Defender": {
            "Metrics": {
                "Passes réceptionnées par 90": "Nombre de passes réceptionnées par 90 minutes",
                "Passes en avant précises, %": "Pourcentage de passes en avant précises",
                "Interceptions PAdj": "Interceptions par 90 minutes (PAdj)",
                "Tacles glissés PAdj": "Tacles glissés par 90 minutes (PAdj)",
                "Fautes par 90": "Nombre de fautes par 90 minutes",
                "Duels aériens par 90": "Nombre de duels aériens par 90 minutes",
                "Duels aériens gagnés, %": "Pourcentage de duels aériens gagnés",
            }
        },
        "Full Wing Back": {
            "Metrics": {
                "Passes réceptionnées par 90": "Nombre de passes réceptionnées par 90 minutes",
                "Passes en avant précises, %": "Pourcentage de passes en avant précises",
                "Interceptions PAdj": "Interceptions par 90 minutes (PAdj)",
                "Tacles glissés PAdj": "Tacles glissés par 90 minutes (PAdj)",
                "Fautes par 90": "Nombre de fautes par 90 minutes",
                "Duels aériens par 90": "Nombre de duels aériens par 90 minutes",
                "Duels aériens gagnés, %": "Pourcentage de duels aériens gagnés",
                "Сentres précises, %": "Pourcentage de centres précis",
                "xA par 90": "Expected Assists par 90 minutes",
            }
        },
    }

    # Define metrics of each template
    templates = {
        "Striker": ['90s', 'Joueur', 'Équipe', 'Place', 'Âge', 'xG par 90', 'Tirs par 90', 'Touches de balle dans la surface de réparation sur 90', 'Actions défensives réussies par 90', 
                    'Duels aériens gagnés par 90', 'Dribbles réussis par 90', 'xG/Tir', 'Passes réceptionnées par 90'],

        "Winger/Attacking Midfielder": ['90s', 'Joueur', 'Équipe', 'Place', 'Âge', 'xG par 90', 'xA par 90', 'Tirs par 90', 'Touches de balle dans la surface de réparation sur 90',
                                         'Сentres précises, %', 'Fautes subies par 90', 'Dribbles réussis par 90', 'Passes vers la surface de réparation précises, %', 'Interceptions PAdj'],

        "Midfielder": ['90s', 'Joueur', 'Équipe', 'Place', 'Âge', 'Passes précises, %', 'Passes progressives par 90', 'xA par 90', 'Dribbles réussis par 90', 'Fautes subies par 90',
                        'Interceptions PAdj', 'Courses progressives par 90'],

        "Defender": ['90s', 'Joueur', 'Équipe', 'Place', 'Âge', 'Passes réceptionnées par 90', 'Passes en avant précises, %', 'Interceptions PAdj', 'Tacles glissés PAdj', 'Fautes par 90', 
                     'Duels aériens par 90', 'Duels aériens gagnés, %'],

        "Full Wing Back": ['90s', 'Joueur', 'Équipe', 'Place', 'Âge', 'Passes réceptionnées par 90', 'Passes en avant précises, %', 'Interceptions PAdj', 'Tacles glissés PAdj',
                            'Fautes par 90', 'Duels aériens par 90', 'Duels aériens gagnés, %', 'Сentres précises, %', 'xA par 90']
    }


    # Streamlit app title
    st.title('Player Similarity App')

    # Sidebar widgets
    selected_player = st.sidebar.selectbox(' Player', df['Joueur'].unique())
    selected_template = st.sidebar.selectbox(' Position Template', list(templates.keys()))
    age_interval = st.sidebar.slider(' Age Interval', min_value=min_age, max_value=max_age, step=1, value=(min_age, max_age))
    nineties_interval = st.sidebar.slider(' 90s Interval', min_value=df['90s'].min(), max_value=df['90s'].max(), step=0.1, value=(df['90s'].min(), df['90s'].max()))
    distance_metric = st.sidebar.radio(' Distance Metric', ['Cosine', 'Euclidean'], index=0)  # 'Cosine' is the default

    default_positions = df[df['Joueur'] == selected_player]['Place'].str.split(', ').tolist()[0]
    selected_positions = st.sidebar.multiselect('Player Positions', default_positions, [])
    selected_leagues = st.sidebar.multiselect(' Player Positions', df['League'].unique(), [])

    # Display metrics and their definitions for the selected template as a guide to users
    if selected_template:
        metrics = templates_info[selected_template]["Metrics"]

        with st.expander("Metrics guide"):
            for metric, definition in metrics.items():
                st.write(f"- {metric}: {definition}")


    # Get selected template's metrics & Allow the user to enter custom metric weights
    template_values = templates[selected_template]
    selected_features = templates[selected_template][5:]

    metric_weights = {}
    st.sidebar.write(f'Custom Metric Weights for {selected_template}:')
    for metric in selected_features:
        weight = st.sidebar.slider(f'{metric}', 0.0, 1.0, 1.0, 0.1)
        metric_weights[metric] = weight

    # Buttons for generating similar players dataframe

    similar_players= None

        
    # Calculate similar players, and cache the result
    similar_players = process_data(df, selected_player, template_values, metric_weights, distance_metric)

    if similar_players is not None:

        # Filter and display similar players using the cached result
        filtered_players = filter_players(similar_players, age_interval, nineties_interval, selected_positions, selected_leagues)
        displayed_df = filtered_players[['Joueur', 'Équipe', 'League', 'Âge', '90s', 'Similarity Percentage']].reset_index(drop=True)[:10]
        
        # Display generated dataframe
        st.write(f'Most Similar Players to {selected_player}:', displayed_df)

        # Save shortlist in csv format
        csv = convert_df(filtered_players[:10])
        st.download_button("Download Shortlist", csv, "shortlist.csv", "text/csv", key='download-csv')

 
        pdf_buffer = generate_pdf_report(selected_player, age_interval, nineties_interval, selected_positions, selected_leagues, displayed_df.to_dict(orient='records'))
        
        # Offer the PDF for download
        st.download_button("Download Report", pdf_buffer, key='download-pdf', file_name="player_similarity_report.pdf", mime="application/pdf")

        # Radar chart to visualise selected player and the one most similar to him
        radar_chart_option = generate_radar_chart_option(df, selected_player, displayed_df['Joueur'][1], selected_features)
        st_echarts(radar_chart_option, height="500px")


 



if __name__ == "__main__":
    main()

