import pandas as pd 
import streamlit as st 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

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



