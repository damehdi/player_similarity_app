import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import streamlit as st

@st.cache_data
def load_data(data_path):
    """
    Load data from a specified data path.

    Parameters:
        data_path (str): The path to the data in Parquet format.

    Returns:
        pd.DataFrame: A Pandas DataFrame containing the loaded data.
    """
    return pd.read_parquet(data_path)

def scale_data(df, template):
    """
    Scale the data in a DataFrame based on a specified template.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        template (list): A list of columns to use as a template for scaling.

    Returns:
        pd.DataFrame: A DataFrame with scaled features.
    """
    not_to_scale = ['90s', 'Joueur', 'Équipe', 'Place', 'Âge']
    df = df[template]
    scaler = StandardScaler()
    scaler.fit(df.drop(not_to_scale, axis=1))
    scaled_features = scaler.transform(df.drop(not_to_scale, axis=1))
    df = pd.DataFrame(scaled_features, columns=df.columns[5:])
    return df

def calculate_distance(X, metric):
    """
    Calculate pairwise distances between data points.

    Parameters:
        X (pd.DataFrame): The input data.
        metric (str): The distance metric to use ('Cosine' or 'Euclidean').

    Returns:
        pd.DataFrame: Pairwise distances between data points.
    """
    if metric == 'Cosine':
        return 1 - pairwise_distances(X, metric='cosine')
    elif metric == 'Euclidean':
        return pairwise_distances(X, metric='euclidean')

def find_similar_players(player_name, df, cosine_similarities):
    """
    Find players similar to a given player based on cosine similarities.

    Parameters:
        player_name (str): The name of the player to find similar players for.
        df (pd.DataFrame): The DataFrame containing player data.
        cosine_similarities (pd.DataFrame): Pairwise cosine similarities between players.

    Returns:
        pd.DataFrame: DataFrame sorted by similarity percentage.
    """
    player_index = df[df['Joueur'] == player_name].index[0]
    similarity_percentages = cosine_similarities[player_index] * 100
    df['Similarity Percentage'] = similarity_percentages
    df = round(df.sort_values(by='Similarity Percentage', ascending=False), 1)
    return df

st.cache_resource
def process_data(df, selected_player, template, metric_weights, distance_metric):
    """
    Process data, find similar players, and perform clustering.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        selected_player (str): The name of the player to find similar players for.
        template (list): A list of columns to use as a template for scaling.
        metric_weights (dict): Custom weights for distance metrics.
        distance_metric (str): The distance metric to use ('Cosine' or 'Euclidean').

    Returns:
        pd.DataFrame: DataFrame containing similar players.
    """
    # Scale the data
    X = scale_data(df, template)

    # Apply custom metric weights
    for metric, weight in metric_weights.items():
        X[metric] *= weight

    # Determine the optimal number of clusters (K) using the silhouette method
    silhouette_scores = []
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=0)
        cluster_assignments = kmeans.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_assignments)
        silhouette_scores.append(silhouette_avg)

    # Find the K value with the highest silhouette score
    optimal_k = np.argmax(silhouette_scores) + 2  # Add 2 to get the actual K value (starting from 2 clusters)

    # Fit K-Means clustering with the chosen K
    kmeans = KMeans(n_clusters=optimal_k, random_state=0)
    cluster_assignments = kmeans.fit_predict(X)

    # Add cluster assignments to the dataframe
    df['cluster'] = cluster_assignments

    # Calculate pairwise cosine similarities
    cosine_similarities = calculate_distance(X, distance_metric)

    # Find similar players for the selected player
    similar_players_df = find_similar_players(selected_player, df, cosine_similarities)

    return similar_players_df

@st.cache
def filter_players(df, age_interval, nineties_interval, selected_positions, selected_leagues):
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
    # Filter based on age_interval
    df = df[(df['Âge'] >= age_interval[0]) & (df['Âge'] <= age_interval[1])]

    # Filter based on nineties_interval
    df = df[(df['90s'] >= nineties_interval[0]) & (df['90s'] <= nineties_interval[1])]

    mask = df['League'].str.contains('|'.join(selected_leagues))

    # Apply the mask to filter the DataFrame
    df = df[mask]

    # Apply position-based filtering
    for position in selected_positions:
        df = df[df['Place'].str.contains(position)]

    return df


def generate_radar_chart_option(df, selected_player, similar_player, selected_features):
    """
    Generate options for a radar chart comparing two players.

    Parameters:
        df (pd.DataFrame): The DataFrame containing player data.
        selected_player (str): The name of the selected player.
        similar_player (str): The name of the similar player.
        selected_features (list): A list of selected features for the radar chart.

    Returns:
        dict: Radar chart options.
    """
    # Filter the DataFrame to get data for the selected player and the most similar player
    selected_player_data = df[df['Joueur'] == selected_player][selected_features].values[0]
    similar_player_data = df[df['Joueur'] == similar_player][selected_features].values[0]

    # Define the indicator names and maximum values based on selected features
    indicators = [{"name": feature, "max": df[feature].max()} for feature in selected_features]

    # Create the radar chart option
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

# Function to convert a DataFrame to CSV format
def convert_df(df):
    """
    Convert a DataFrame to CSV format.

    Parameters:
        df (pd.DataFrame): The DataFrame to be converted.

    Returns:
        bytes: The CSV data encoded in UTF-8.
    """
    return df.to_csv(index=False).encode('utf-8')

from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet

def generate_pdf_report(selected_player, age_interval, nineties_interval, selected_positions, selected_leagues, similar_players):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)

    styles = getSampleStyleSheet()
    story = []

    # Add content to the PDF
    title = Paragraph("Player Similarity Report", styles["Title"])
    story.append(title)
    story.append(Spacer(1, 12))

    # Filtering parameters
    filter_info = [
        f"Selected Player: {selected_player}",
        f"Age Interval: {age_interval[0]} - {age_interval[1]}",
        f"90s Interval: {nineties_interval[0]} - {nineties_interval[1]}",
        f"Selected Positions: {', '.join(selected_positions)}",
        f"Selected Leagues: {', '.join(selected_leagues)}",
    ]

    filter_text = Paragraph("<br/>".join(filter_info), styles["Normal"])
    story.append(filter_text)
    story.append(Spacer(1, 12))

    # Similar players information
    similar_players_title = Paragraph("Most Similar Players to:", styles["Heading2"])
    story.append(similar_players_title)

    # Create a numbered list of similar players
    similar_players_list = ListFlowable(
        [ListItem(Paragraph(f"{player['Joueur']} ({player['Équipe']} - {player['League']}) [{player['Similarity Percentage']}%]", styles["Normal"])) for player in similar_players],
        bulletType='1', start=1
    )
    story.append(similar_players_list)

    # Build the PDF
    doc.build(story)

    # Reset the buffer position to the beginning
    buffer.seek(0)

    return buffer







