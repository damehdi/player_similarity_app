import pandas as pd 
import streamlit as st 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from fpdf import FPDF


@st.cache_data
def load_data(data_path):
    return pd.read_parquet(data_path)

def create_sidebar_widgets(df: pd.DataFrame):

    # Extract minimum and maximum age 
    min_age = df['Âge'].min()
    max_age = df['Âge'].max()

    # Define sidebar widgets
    selected_player = st.sidebar.selectbox('Player', df['Joueur'].unique())
    selected_template = st.sidebar.selectbox('Position Template', list(templates.keys()))
    age_interval = st.sidebar.slider('Age Interval', min_value=min_age, max_value=max_age, step=1, value=(min_age, max_age))
    nineties_interval = st.sidebar.slider('90s Interval', min_value=df['90s'].min(), max_value=df['90s'].max(), step=0.1, value=(df['90s'].min(), df['90s'].max()))
    distance_metric = st.sidebar.radio('Distance Metric', ['Cosine', 'Euclidean'], index=0)  # 'Cosine' is the default

    default_positions = df[df['Joueur'] == selected_player]['Place'].str.split(', ').tolist()[0]
    selected_positions = st.sidebar.multiselect('Player Positions', default_positions, [])
    selected_leagues = st.sidebar.multiselect('Player Leagues', df['League'].unique(), [])
    
    return selected_player, selected_template, age_interval, nineties_interval, distance_metric, selected_positions, selected_leagues

def scale_data(data: pd.DataFrame, columns_to_keep: list):
    columns_to_exclude = ['90s', 'Joueur', 'Équipe', 'Place', 'Âge']
    data_subset = data[columns_to_keep]
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data_subset.drop(columns=columns_to_exclude))
    
    scaled_df = pd.DataFrame(scaled_features, columns=data_subset.columns[5:])
    return scaled_df

def calculate_distance(X, metric):
    valid_metrics = {'cosine', 'euclidean'}

    if metric not in valid_metrics:
        raise ValueError(f"Unsupported distance metric: {metric}")

    return 1 - pairwise_distances(X, metric=metric) if metric == 'cosine' else pairwise_distances(X, metric=metric)

def find_similar_players(player, df, cosine_similarities):
    player_index = df.index[df['Joueur'] == player][0]
    df['Similarity Percentage'] = cosine_similarities[player_index] * 100
    df= df.sort_values(by='Similarity Percentage', ascending=False).round(1)
    return df

def process_data(df, player, template, metrics_weights, distance_metric ):
    # Scaling data
    X= scale_data(df, template)

    for metric, weight in metrics_weights.items():
        X[metric] *= weight 
    
    cosine_similarities= calculate_distance(X, distance_metric)
    similar_players_df= find_similar_players(player, df, cosine_similarities)
    return similar_players_df

import pandas as pd

def filter_data(df, age_interval, nineties_interval, selected_positions, selected_leagues):
    """
    Filter players based on specified criteria.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing player data.
        age_interval (tuple): A tuple specifying the age range to filter.
        nineties_interval (tuple): A tuple specifying the '90s range to filter.
        selected_positions (list): A list of selected player positions to filter.
        selected_leagues (list): A list of selected leagues to filter.

    Returns:
        pd.DataFrame: DataFrame containing filtered players.
    """
    age_filter = (df['Âge'] >= age_interval[0]) & (df['Âge'] <= age_interval[1])

    # Filter based on nineties_interval & leagues and positions
    nineties_filter = (df['90s'] >= nineties_interval[0]) & (df['90s'] <= nineties_interval[1])
    league_mask = df['League'].str.contains('|'.join(selected_leagues))
    position_mask = df['Place'].str.contains('|'.join(selected_positions))

    # Combine all filters
    filtered_df = df[age_filter & nineties_filter & league_mask & position_mask]

    return filtered_df

def convert_data(df):
    """
    Convert a DataFrame to CSV format.
    """
    return df.to_csv(index=False).encode('utf-8')

def generate_radar_chart_option(data, selected_player, similar_player, selected_features):

    # filter the DataFrame to get data for the selected player and the most similar player
    selected_player_data = data[data['Joueur'] == selected_player][selected_features].values[0]
    similar_player_data = data[data['Joueur'] == similar_player][selected_features].values[0]

    # define the indicator names and maximum values based on selected features and creating the radar chart option
    indicators = [{"name": feature, "max": data[feature].max()} for feature in selected_features]
    option = {
        "title": {"text": "Radar Chart Comparison"},
        "legend": {"data": [selected_player, similar_player]},
        "radar": {"indicator": indicators},
        "series": [
            {
                "name": f"{selected_player} vs {similar_player}",
                "type": "radar",
                "data": [
                    {"value": selected_player_data.tolist(), "name": selected_player},
                    {"value": similar_player_data.tolist(), "name": similar_player},
                ],
            }
        ],
    }
    return option

def generate_pdf_report(player, age_interval, nineties_interval, selected_positions, selected_leagues, similar_players_df):
    
    class PDF(FPDF):
        def header(self):
            pass  

        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, 'Page %s' % self.page_no(), 0, 0, 'C')

    pdf = PDF()
    pdf.add_page()
    
    pdf.set_font('Arial', '', 12)
    
    # background image
    pdf.image('DOC01.jpg', x=0, y=0, w=210)

    # pdf content
    pdf.set_xy(20, 30) 
    
    pdf.set_font('Arial', 'B', 19)
    pdf.cell(0, 10, 'Rapport de Similarité des Joueurs', ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Filtres Sélectionnées:', ln=True,)

    pdf.set_font('Arial', '', 12)
    pdf.multi_cell(0, 10, 'Joueur Sélectionné: {}\n'
                          'Intervalle d\'âge: {} - {}\n'
                          'Intervalle des années 90: {} - {}\n'
                          'Positions Sélectionnées: {}\n'
                          'Ligues Sélectionnées: {}'.format(player,
                                                       age_interval[0], age_interval[1],
                                                       nineties_interval[0], nineties_interval[1],
                                                       ', '.join(selected_positions),
                                                       ', '.join(selected_leagues)))
    pdf.ln(10)
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Joueurs les Plus Similaires:', ln=True,)
    pdf.ln(10)

    # display similar players dataframe in a table format
    pdf.set_fill_color(200, 220, 255)  
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(20, 10, 'Rank', 1, 0, 'C', 1)
    pdf.cell(60, 10, 'Joueur', 1, 0, 'C', 1)
    pdf.cell(40, 10, 'Équipe', 1, 0, 'C', 1)
    pdf.cell(40, 10, 'Ligue', 1, 0, 'C', 1)
    pdf.cell(30, 10, 'Similarité (%)', 1, 1, 'C', 1)

    pdf.set_font('Arial', '', 12)
    for i, row in similar_players_df.iterrows():
        pdf.cell(20, 10, str(i + 1), 1)
        pdf.cell(60, 10, row['Joueur'], 1)
        pdf.cell(40, 10, row['Équipe'], 1)
        pdf.cell(40, 10, row['League'], 1)
        pdf.cell(30, 10, str(row['Similarity Percentage']), 1, 1)

    # save pdf
    pdf.output('player_similarity_report.pdf')