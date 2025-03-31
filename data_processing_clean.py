import requests, json
from pprint import pprint
import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas()
import warnings
warnings.filterwarnings("ignore")

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def fetch_data():
    """Fetches the main data from the FPL API."""
    r = requests.get('https://fantasy.premierleague.com/api/bootstrap-static/', verify=False).json()
    return r

def create_players_df(data):
    """Creates the players DataFrame from the raw data."""
    players = pd.json_normalize(data['elements'])
    players.drop(columns=[
        'can_transact', 'can_select', 'cost_change_event', 'cost_change_event_fall', 'cost_change_start',
        'cost_change_start_fall', 'dreamteam_count', 'in_dreamteam', 'news', 'news_added', 'photo', 'special',
        'squad_number', 'region', 'influence_rank', 'influence_rank_type', 'creativity_rank',
        'creativity_rank_type', 'threat_rank', 'threat_rank_type', 'ict_index_rank', 'ict_index_rank_type',
        'corners_and_indirect_freekicks_text', 'direct_freekicks_text', 'penalties_text', 'now_cost_rank',
        'now_cost_rank_type', 'form_rank_type', 'points_per_game_rank_type', 'selected_rank',
        'selected_rank_type', 'removed', 'chance_of_playing_next_round', 'chance_of_playing_this_round',
        'event_points', 'transfers_in', 'transfers_in_event', 'transfers_out', 'transfers_out_event',
        'team_join_date', 'has_temporary_code', 'opta_code'
    ], inplace=True)
    return players

def create_teams_df(data):
    """Creates the teams DataFrame from the raw data."""
    teams = pd.json_normalize(data['teams'])
    teams.drop(columns=[
        'draw', 'loss', 'played', 'points', 'position', 'team_division',
        'unavailable', 'win', 'pulse_id'
    ], inplace=True)
    return teams

def create_positions_df(data):
    """Creates the positions DataFrame from the raw data."""
    positions = pd.json_normalize(data['element_types'])
    positions.drop(columns=[
        'plural_name', 'plural_name_short', 'squad_select', 'squad_min_select',
        'squad_max_select', 'squad_min_play', 'squad_max_play', 'ui_shirt_specific',
        'sub_positions_locked', 'element_count'
    ], inplace=True)
    return positions

def merge_data(players, teams, positions):
    """Merges the players, teams, and positions data into a single DataFrame."""
    df = pd.merge(players, teams, left_on='team', right_on='id')
    df = df.merge(positions, left_on='element_type', right_on='id')
    df = df.rename(
        columns={'name': 'team_name', 'singular_name': 'position_name',
                 'singular_name_short': 'position', 'id_y': 'team_id', 'id_x': 'player_id'}
    )
    return df

# Run the functions
data = fetch_data()
players = create_players_df(data)
teams = create_teams_df(data)
positions = create_positions_df(data)
df = merge_data(players, teams, positions)

def fetch_player_summary(player_id):
    """Fetches the summary data for a given player ID."""
    r = requests.get('https://fantasy.premierleague.com/api/element-summary/' + str(player_id)/ + '/',verify=False).json()
    return r

def get_fixtures(player_id):
    """Get all future fixtures for a given player ID."""
    r = requests.get('https://fantasy.premierleague.com/api/element-summary/' + str(player_id) + '/',verify=False).json()
    df = pd.json_normalize(r['fixtures'])
    df['player_id'] = player_id
    return df.rename(columns={'event': 'GW','id':'fixture'})

def get_all_fixtures(df):
    """Get future fixtures for all players."""
    fixtures = df['player_id'].progress_apply(get_fixtures)
    fixtures = pd.concat(list(fixtures), ignore_index=True)
    return fixtures


def get_gameweek_history(player_id):
    """Get all gameweek info for a given player ID."""
    r = requests.get('https://fantasy.premierleague.com/api/element-summary/' + str(player_id) + '/').json()
    df = pd.json_normalize(r['history'])
    return df

def get_all_gameweek_history(df):
    """Get gameweek history for all players."""
    history = df['player_id'].progress_apply(get_gameweek_history)
    history = pd.concat(list(history), ignore_index=True)
    return history

def clean_gameweek_history(history):
    """Clean and transform the gameweek history DataFrame."""
    # Rename columns to standardize and merge later
    history.rename(columns={'element': 'player_id', 'was_home': 'is_home', 'round': 'GW'}, inplace=True)
    cols_to_drop = ['value', 'transfers_balance', 'selected', 'transfers_in', 'transfers_out']
    history.drop(columns=cols_to_drop, inplace=True)
    return history

def merge_history_with_player_info(df, history):
    """Merge player information with gameweek history."""
    history = df[[
        'player_id', 'web_name', 'team_id', 'position', 'strength_overall_home',
        'strength_overall_away', 'strength_attack_home', 'strength_attack_away',
        'strength_defence_home', 'strength_defence_away',
        'corners_and_indirect_freekicks_order', 'direct_freekicks_order', 'penalties_order'
    ]].merge(history, on='player_id')

    # Drop unnecessary columns
    history.drop(columns=['modified', 'influence', 'creativity', 'threat', 'ict_index'], inplace=True)

    # Rename columns to make them consistent
    history.rename(columns={'web_name': 'player'}, inplace=True)

    # Convert numeric columns
    cols_to_convert = ['expected_goals', 'expected_assists', 'expected_goal_involvements', 'expected_goals_conceded']
    history[cols_to_convert] = history[cols_to_convert].apply(pd.to_numeric, errors='coerce')

    return history

# Example usage
print("Getting all fixtures...")
fixtures = get_all_fixtures(df)
print("Fixtures obtained!")

print("Getting all gameweek history...")
history = get_all_gameweek_history(df)
print("History obtained!")

print("Cleaning gameweek history...")
history = clean_gameweek_history(history)
print("History cleaned!")

print("Merging history with player info...")
history = merge_history_with_player_info(df, history)
print("History merged!")

import numpy as np

def extract_team_strengths(df):
    """Extract unique team strength data from the main dataframe."""
    team_strengths = df[['team_id', 'strength_overall_home', 'strength_overall_away',
                         'strength_attack_home', 'strength_attack_away',
                         'strength_defence_home', 'strength_defence_away']].drop_duplicates()
    team_strengths.set_index('team_id', inplace=True)
    return team_strengths

def calculate_fixtures_metrics(fixtures, team_strengths):
    """Calculate opponent and team strength metrics for fixtures."""
    # Determine opponent based on home or away
    fixtures["opponent"] = fixtures.apply(
        lambda row: row["team_a"] if row["is_home"] else row["team_h"], axis=1
    )

    # Determine player's own team based on whether they are at home or away
    fixtures['team_id'] = np.where(fixtures['is_home'], fixtures['team_h'], fixtures['team_a'])

    # Opponent strength metrics
    fixtures['opponent_strength_overall'] = np.where(
        fixtures['is_home'],
        fixtures['opponent'].map(team_strengths['strength_overall_away']),
        fixtures['opponent'].map(team_strengths['strength_overall_home'])
    )

    fixtures['opponent_strength_attack'] = np.where(
        fixtures['is_home'],
        fixtures['opponent'].map(team_strengths['strength_attack_away']),
        fixtures['opponent'].map(team_strengths['strength_attack_home'])
    )

    fixtures['opponent_strength_defence'] = np.where(
        fixtures['is_home'],
        fixtures['opponent'].map(team_strengths['strength_defence_away']),
        fixtures['opponent'].map(team_strengths['strength_defence_home'])
    )

    # Team strength metrics
    fixtures['team_strength_overall'] = np.where(
        fixtures['is_home'],
        fixtures['team_id'].map(team_strengths['strength_overall_home']),
        fixtures['team_id'].map(team_strengths['strength_overall_away'])
    )

    fixtures['team_strength_attack'] = np.where(
        fixtures['is_home'],
        fixtures['team_id'].map(team_strengths['strength_attack_home']),
        fixtures['team_id'].map(team_strengths['strength_attack_away'])
    )

    fixtures['team_strength_defence'] = np.where(
        fixtures['is_home'],
        fixtures['team_id'].map(team_strengths['strength_defence_home']),
        fixtures['team_id'].map(team_strengths['strength_defence_away'])
    )

    return fixtures

def calculate_history_metrics(history, team_strengths):
    """Calculate team and opponent strength metrics for history."""
    # Team Strength Metrics
    history['team_strength_overall'] = np.where(
        history['is_home'],
        history['team_id'].map(team_strengths['strength_overall_home']),
        history['team_id'].map(team_strengths['strength_overall_away'])
    )

    history['team_strength_attack'] = np.where(
        history['is_home'],
        history['team_id'].map(team_strengths['strength_attack_home']),
        history['team_id'].map(team_strengths['strength_attack_away'])
    )

    history['team_strength_defence'] = np.where(
        history['is_home'],
        history['team_id'].map(team_strengths['strength_defence_home']),
        history['team_id'].map(team_strengths['strength_defence_away'])
    )

    # Opponent Strength Metrics
    history['opponent_strength_overall'] = np.where(
        history['is_home'],
        history['opponent_team'].map(team_strengths['strength_overall_away']),
        history['opponent_team'].map(team_strengths['strength_overall_home'])
    )

    history['opponent_strength_attack'] = np.where(
        history['is_home'],
        history['opponent_team'].map(team_strengths['strength_attack_away']),
        history['opponent_team'].map(team_strengths['strength_attack_home'])
    )

    history['opponent_strength_defence'] = np.where(
        history['is_home'],
        history['opponent_team'].map(team_strengths['strength_defence_away']),
        history['opponent_team'].map(team_strengths['strength_defence_home'])
    )

    # Rename to match fixtures for concatenation
    history.rename(columns={'opponent_team': 'opponent'}, inplace=True)
    return history

def concatenate_fixtures_and_history(history, fixtures):
    """Concatenate past and future fixtures along player_id."""
    final_df = pd.concat([history.reset_index().set_index("player_id"), fixtures.set_index("player_id")], axis=0, ignore_index=False)
    return final_df

# Run functions to process the data
print("Extracting team strengths...")
team_strengths = extract_team_strengths(df)

print("Calculating fixture metrics...")
fixtures = calculate_fixtures_metrics(fixtures, team_strengths)

print("Calculating history metrics...")
history = calculate_history_metrics(history, team_strengths)

print("Concatenating fixtures and history...")
final_df = concatenate_fixtures_and_history(history, fixtures)
print("Data processing complete!")

def fill_na_details(final_df):
    """Fill NA values for necessary details."""
    columns_to_fill = [
        'index', 'player', 'team_id', 'position',
        'strength_overall_home', 'strength_overall_away',
        'strength_attack_home', 'strength_attack_away',
        'strength_defence_home', 'strength_defence_away',
        'corners_and_indirect_freekicks_order',
        'direct_freekicks_order', 'penalties_order'
    ]
    final_df[columns_to_fill] = final_df.groupby('player_id')[columns_to_fill].transform(lambda x: x.ffill().bfill())
    return final_df

def fill_difficulty(final_df):
    """Fill missing difficulty values based on known opponents."""
    difficulty_map = final_df.dropna(subset=['difficulty']).set_index('opponent')['difficulty'].to_dict()
    final_df['difficulty'] = final_df['difficulty'].fillna(final_df['opponent'].map(difficulty_map))
    return final_df

def cleanup_final_df(final_df):
    """Cleanup and prepare the final DataFrame."""
    final_df.reset_index(inplace=True)
    final_df.drop(columns=['index','code', 'team_h', 'team_a',
                           'finished', 'provisional_start_time', 'event_name'], inplace=True)

    # Drop redundant team strength columns
    cols_to_drop = ['strength_overall_home', 'strength_overall_away',
                    'strength_attack_home', 'strength_attack_away',
                    'strength_defence_home', 'strength_defence_away']
    final_df.drop(columns=cols_to_drop, inplace=True)
    return final_df

def calculate_relative_strength(final_df):
    """Calculate relative strength metrics."""
    final_df['relative_strength_overall'] = final_df['team_strength_overall'] - final_df['opponent_strength_overall']
    final_df['relative_strength_attack'] = final_df['team_strength_attack'] - final_df['opponent_strength_defence']
    final_df['relative_strength_defence'] = final_df['team_strength_defence'] - final_df['opponent_strength_attack']
    return final_df

def process_kickoff_time(final_df):
    """Process kickoff time and categorize it."""
    final_df['kickoff_time'] = pd.to_datetime(final_df['kickoff_time'], errors='coerce')
    final_df['kickoff_time'] = final_df['kickoff_time'].dt.hour
    return final_df

def categorize_kickoff(kickoff):
    """Categorize the kickoff time."""
    if kickoff < 14:
        return 'early'
    elif 14 <= kickoff < 18:
        return 'afternoon'
    else:
        return 'late'

def encode_kickoff_and_location(final_df):
    """Encode kickoff categories and home/away indicator."""
    # Categorize kickoff times
    final_df['kickoff_category'] = final_df['kickoff_time'].apply(categorize_kickoff)
    final_df['is_home'] = final_df['is_home'].astype(int)  # 1=Home, 0=Away

    # Find the most common kickoff category
    most_common_category = final_df['kickoff_category'].mode()[0]

    # One-hot encode kickoff categories
    final_df = pd.get_dummies(final_df, columns=['kickoff_category'], prefix='kickoff')

    # Drop the most common kickoff category column
    final_df.drop(columns=[f'kickoff_{most_common_category}'], inplace=True)
    return final_df

# Applying the functions to the final DataFrame
print("Filling NA details...")
final_df = fill_na_details(final_df)

print("Filling difficulty values...")
final_df = fill_difficulty(final_df)

print("Cleaning up final DataFrame...")
final_df = cleanup_final_df(final_df)

print("Calculating relative strength metrics...")
final_df = calculate_relative_strength(final_df)

print("Processing kickoff times...")
final_df = process_kickoff_time(final_df)

print("Encoding kickoff and home/away indicators...")
final_df = encode_kickoff_and_location(final_df)

print("Data preprocessing complete!")

def fill_with_rolling_average(df, gw_threshold=29): #eventually use an if statement to check current gw_threshold based on date/time
    """
    Impute missing values for statistics by filling with a rolling average.
    This assumes players continue at the level of their current (3/5 week?) average.
    """
    cols_to_fill = [
        'minutes', 'goals_scored', 'assists', 'clean_sheets', 'goals_conceded', 'starts',
        'expected_goals', 'expected_assists', 'saves', 'expected_goals_conceded'
    ]  # ADD MORE IF NECESSARY

    df = df.copy()

    # Convert columns to float for processing
    for col in cols_to_fill:
        df[col] = df[col].astype(float)

    # Iterate over players
    for player in df['player_id'].unique():
        player_mask = df['player_id'] == player
        player_df = df[player_mask].copy()

        for gw in range(int(gw_threshold) + 1, int(df['GW'].max()) + 1):
          #
            past_values = df[(df['player_id'] == player) & (df['GW'] < gw)]

            for col in cols_to_fill:
                if not past_values.empty:
                    # Compute rolling mean with new values
                    rolling_avg = past_values[col].fillna(0).rolling(window=5, min_periods=2).mean().iloc[-1]

                    # Assign the averages
                    df.loc[(df['player_id'] == player) & (df['GW'] == gw), col] = rolling_avg

    # Replace nans
    df.fillna(0, inplace=True)

    return df

# Applying the function to the final DataFrame
print("Filling missing values with rolling averages...")
final_df = fill_with_rolling_average(final_df)
print("Rolling average imputation completed!")

def calculate_rolling_features(final_df):
    """Calculate various rolling average features for the dataframe."""
    final_df = final_df.sort_values(['player_id', 'GW'])

    # Rolling averages for goals
    final_df['rolling_goals_3'] = final_df.groupby('player_id')['goals_scored'].shift(1).rolling(window=3, min_periods=1).mean()
    final_df['rolling_goals_10'] = final_df.groupby('player_id')['goals_scored'].shift(4).rolling(window=10, min_periods=5).mean()

    # Rolling averages for assists
    final_df['rolling_assists_3'] = final_df.groupby('player_id')['assists'].shift(1).rolling(window=3, min_periods=1).mean()
    final_df['rolling_assists_10'] = final_df.groupby('player_id')['assists'].shift(1).rolling(window=10, min_periods=5).mean()

    # Rolling averages for expected goals (xG)
    final_df['rolling_xg_3'] = final_df.groupby('player_id')['expected_goals'].shift(1).rolling(window=3, min_periods=1).mean()
    final_df['rolling_xg_10'] = final_df.groupby('player_id')['expected_goals'].shift(1).rolling(window=10, min_periods=5).mean()

    # Rolling averages for expected assists (xA)
    final_df['rolling_xa_3'] = final_df.groupby('player_id')['expected_assists'].shift(1).rolling(window=3, min_periods=1).mean()
    final_df['rolling_xa_10'] = final_df.groupby('player_id')['expected_assists'].shift(1).rolling(window=10, min_periods=5).mean()

    # Rolling averages for minutes
    final_df['rolling_mins_3'] = final_df.groupby('player_id')['minutes'].shift(1).rolling(window=3, min_periods=3).mean() / 90
    final_df['rolling_mins_10'] = final_df.groupby('player_id')['minutes'].shift(1).rolling(window=10, min_periods=5).mean() / 90

    # Finishing efficiency
    final_df['finishing_efficiency'] = final_df.groupby('player_id')['goals_scored'].shift(1) / final_df.groupby('player_id')['expected_goals'].shift(1)
    final_df['finishing_efficiency'].replace([float('inf'), -float('inf')], 0, inplace=True)

    # Teammate clinicalness
    final_df['teammate_clinicalness'] = final_df.groupby('player_id')['assists'].shift(1) / final_df.groupby('player_id')['expected_assists'].shift(1)
    final_df['teammate_clinicalness'].replace([float('inf'), -float('inf')], 0, inplace=True)

    # Defensive efficiency
    final_df['defensive_efficiency'] = final_df.groupby('player_id')['goals_conceded'].shift(1) / final_df.groupby('player_id')['expected_goals_conceded'].shift(1)
    final_df['defensive_efficiency'].replace([float('inf'), -float('inf')], 0, inplace=True)

    # Rolling averages for expected goals (xG)
    final_df['rolling_xgc_3'] = final_df.groupby('player_id')['expected_goals_conceded'].shift(1).rolling(window=3, min_periods=1).mean()
    final_df['rolling_xgc_10'] = final_df.groupby('player_id')['expected_goals_conceded'].shift(1).rolling(window=10, min_periods=5).mean()

    return final_df


def calculate_rolling_efficiency(final_df):
    """Calculate rolling efficiency features."""
    # Finishing efficiency
    final_df['rolling_finishing_efficiency_3'] = final_df.groupby('player_id')['finishing_efficiency'].shift(1).rolling(window=3, min_periods=1).mean()
    final_df['rolling_finishing_efficiency_10'] = final_df.groupby('player_id')['finishing_efficiency'].shift(1).rolling(window=10, min_periods=5).mean()

    # Teammate clinicalness
    final_df['rolling_teammate_clinicalness_3'] = final_df.groupby('player_id')['teammate_clinicalness'].shift(1).rolling(window=3, min_periods=1).mean()
    final_df['rolling_teammate_clinicalness_10'] = final_df.groupby('player_id')['teammate_clinicalness'].shift(1).rolling(window=10, min_periods=5).mean()

    # Defensive efficiency
    final_df['rolling_defensive_efficiency_3'] = final_df.groupby('player_id')['defensive_efficiency'].shift(1).rolling(window=3, min_periods=1).mean()
    final_df['rolling_defensive_efficiency_10'] = final_df.groupby('player_id')['defensive_efficiency'].shift(1).rolling(window=10, min_periods=5).mean()

    return final_df


def calculate_rolling_team_metrics(final_df):
    """Calculate rolling team metrics."""
    # Clean sheets
    final_df['rolling_clean_sheets_3'] = final_df.groupby('player_id')['clean_sheets'].shift(1).rolling(window=3, min_periods=1).mean()
    final_df['rolling_clean_sheets_10'] = final_df.groupby('player_id')['clean_sheets'].shift(1).rolling(window=10, min_periods=5).mean()

    # Saves for goalkeepers
    final_df['rolling_saves_3'] = final_df.groupby('player_id')['saves'].shift(1).rolling(window=3, min_periods=1).mean()
    final_df['rolling_saves_10'] = final_df.groupby('player_id')['saves'].shift(1).rolling(window=10, min_periods=5).mean()

    return final_df



def calculate_opponent_rolling_goals_conceded(df, windows=[3, 10]):
    """Calculate the rolling goals conceded by the opponent for multiple window sizes."""
    for window in windows:
        col_name = f'opponent_rolling_goals_conceded_{window}'
        df[col_name] = 0.0  #

        # Loop through each row in the dataset and calculate rolling goals conceded for the opponent
        for idx, row in df.iterrows():
            opponent_team_id = row['opponent']  # Get the opponent team ID for the  player

            # Get the opponent's data up to the current gameweek, then group by team and get max goals conceded
            opponent_data = df[(df['team_id'] == opponent_team_id) & (df['GW'] < row['GW'])]

            if not opponent_data.empty:
                # Group by the opponent team and get the max goals conceded for each gameweek
                opponent_max_goals_conceded = opponent_data.groupby('GW')['goals_conceded'].max()

                # Get the rolling mean of the max goals conceded for the past 'window' gameweeks
                rolling_goals_conceded = opponent_max_goals_conceded.rolling(window=window, min_periods=1).mean()

                # Store the rolling goals conceded
                df.at[idx, col_name] = rolling_goals_conceded.iloc[-1]

    return df


def calculate_opponent_rolling_expected_goals_conceded(df, windows=[3, 10]):
    """Calculate the rolling expected goals conceded by the opponent for multiple window sizes."""
    for window in windows:
        col_name = f'opponent_rolling_goals_conceded_{window}'
        df[col_name] = 0.0
        # Loop through each row in the dataset and calculate rolling goals conceded for the opponent
        for idx, row in df.iterrows():
            opponent_team_id = row['opponent']  # Get the opponent team ID for the  player

            # Get the opponent's data up to the current gameweek, then group by team and get max goals conceded
            opponent_data = df[(df['team_id'] == opponent_team_id) & (df['GW'] < row['GW'])]

            if not opponent_data.empty:
                # Group by the opponent team and get the max goals conceded for each gameweek
                opponent_max_goals_conceded = opponent_data.groupby('GW')['expected_goals_conceded'].max()

                # Calculate the rolling mean of the max goals conceded for the past 'window' gameweeks
                rolling_goals_conceded = opponent_max_goals_conceded.rolling(window=window, min_periods=1).mean()

                # Store the rolling goals conceded for this row
                df.at[idx, col_name] = rolling_goals_conceded.iloc[-1]

    return df


def add_binary_features(final_df):
    """Add binary features for set-piece roles."""
    final_df['is_penalty_taker'] = final_df['penalties_order'].apply(lambda x: 1 if x == 1 else 0)
    final_df['is_corner_taker'] = final_df['corners_and_indirect_freekicks_order'].apply(lambda x: 1 if x == 1 else 0)
    final_df['is_free_kick_taker'] = final_df['direct_freekicks_order'].apply(lambda x: 1 if x == 1 else 0)

    # Drop original columns after creating binary features
    final_df.drop(columns=[
        'corners_and_indirect_freekicks_order', 'direct_freekicks_order', 'penalties_order'
    ], inplace=True)

    return final_df


# Applying the functions
print("Calculating rolling features...")
final_df = calculate_rolling_features(final_df)

print("Calculating rolling efficiency...")
final_df = calculate_rolling_efficiency(final_df)

print("Calculating rolling team metrics...")
final_df = calculate_rolling_team_metrics(final_df)

print("Calculating opponent rolling goals conceded...")
final_df = calculate_opponent_rolling_goals_conceded(final_df)

print("Adding binary features...")
final_df = add_binary_features(final_df)

def get_team_goals(row):
    """
    Calculate team goals scored based on whether the player is playing at home or away.
    """
    # If the player is playing at home, use 'team_h_score'; otherwise, use 'team_a_score'
    return row['team_h_score'] if row['is_home'] == 1 else row['team_a_score']

def calculate_team_goals(final_df):
    """
    Add team goals scored and calculate rolling averages for team goals.
    """
    # Add a new column to calculate team goals scored for each row
    final_df['team_goals_scored'] = final_df.apply(get_team_goals, axis=1)

    # Calculate rolling averages for team goals scored
    final_df['rolling_team_goals_3'] = final_df.groupby('player_id')['team_goals_scored'].shift(1).rolling(window=3, min_periods=1).mean()
    final_df['rolling_team_goals_10'] = final_df.groupby('player_id')['team_goals_scored'].shift(1).rolling(window=10, min_periods=5).mean()

    return final_df

# Applying the function to the final DataFrame
print("Calculating team goals scored and rolling averages...")
final_df = calculate_team_goals(final_df)
print("Team goals and rolling averages calculated!")

# List of features you want to compute rolling averages for
columns_to_average = [
    'goals_scored', 'assists', 'expected_goals', 'expected_assists',
    'minutes', 'finishing_efficiency', 'teammate_clinicalness',
    'defensive_efficiency', 'clean_sheets', 'saves','goals_conceded','expected_goals_conceded',
    'team_goals_scored','finishing_efficiency','teammate_clinicalness'
]

# Function to calculate rolling averages for each specified column based on all previous gameweeks
def calculate_rolling_averages(df, columns):
    # Loop through each column to calculate rolling average based on previous gameweeks
    for col in columns:
        # Apply the rolling average for each player, using all previous gameweeks (not including current gameweek)
        df[f'season_avg_{col}'] = df.groupby('player_id').apply(
            lambda group: group.apply(
                lambda row: group.loc[group['GW'] < row['GW'], col].mean() if row['GW'] > 1 else 0, axis=1
            )
        ).reset_index(level=0, drop=True)  # Reset index to ensure proper alignment

    return df


# Apply the function to calculate rolling averages for all specified columns
final_df = calculate_rolling_averages(final_df, columns_to_average)




#  per 90 minute features
def generate_per_90_features(df):
    df['rolling_goals_per_90_3'] = (df['rolling_goals_3'] / df['rolling_mins_3']) * 90
    df['rolling_goals_per_90_10'] = (df['rolling_goals_10'] / df['rolling_mins_10']) * 90
    df['season_goals_per_90'] = (df['season_avg_goals_scored'] / df['season_avg_minutes']) * 90

    df['rolling_xg_per_90_3'] = (df['rolling_xg_3'] / df['rolling_mins_3']) * 90
    df['rolling_xg_per_90_10'] = (df['rolling_xg_10'] / df['rolling_mins_10']) * 90
    df['season_xg_per_90'] = (df['season_avg_expected_goals'] / df['season_avg_minutes']) * 90

    df['rolling_assists_per_90_3'] = (df['rolling_assists_3'] / df['rolling_mins_3']) * 90
    df['rolling_assists_per_90_10'] = (df['rolling_assists_10'] / df['rolling_mins_10']) * 90
    df['season_assists_per_90'] = (df['season_avg_assists'] / df['season_avg_minutes']) * 90

    df['rolling_xa_per_90_3'] = (df['rolling_xa_3'] / df['rolling_mins_3']) * 90
    df['rolling_xa_per_90_10'] = (df['rolling_xa_10'] / df['rolling_mins_10']) * 90
    df['season_xa_per_90'] = (df['season_avg_expected_assists'] / df['season_avg_minutes']) * 90

    return df

#  interactive feature
def generate_interaction_features(df):
    df['scoring_chances_3'] = df['rolling_xg_per_90_3'] * df['rolling_finishing_efficiency_3']
    df['scoring_chances_10'] = df['rolling_xg_per_90_10'] * df['rolling_finishing_efficiency_10']

    df['assist_chances_3'] = df['rolling_xa_per_90_3'] * df['rolling_teammate_clinicalness_3']
    df['assist_chances_10'] = df['rolling_xa_per_90_10'] * df['rolling_teammate_clinicalness_10']

    df['team_goal_involvement_3'] = (df['rolling_goals_3'] + df['rolling_assists_3']) / df['rolling_team_goals_3'] * 100
    df['team_goal_involvement_10'] = (df['rolling_goals_10'] + df['rolling_assists_10']) / df['rolling_team_goals_10'] * 100

    df['opponent_goal_concession_3'] = df['opponent_rolling_goals_conceded_3'] * df['relative_strength_attack']
    df['opponent_goal_concession_10'] = df['opponent_rolling_goals_conceded_10'] * df['relative_strength_attack']

    return df

final_df = generate_per_90_features(final_df)
print("Generated per 90 features")

final_df = generate_interaction_features(final_df)
print("Generated interaction features")

print("Data collection complete")

def save_final_data(df, filename='final_data.csv'):
    """Save the final processed DataFrame to a CSV file."""
    if not isinstance(df, pd.DataFrame):
        print("Error: The final_df is not a DataFrame")
        return
    if df.empty:
        print("Warning: The DataFrame is empty. No data to save.")
        return

    # Ensure columns are string type to avoid issues during save
    df.columns = [str(col) for col in df.columns]

    # Save to CSV
    try:
        df.to_csv(filename, index=False)
        print(f"✅ Data saved to {filename}")
    except Exception as e:
        print(f"Error saving data: {e}")

save_final_data(final_df)


print("Pipeline complete")
