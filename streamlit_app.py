import streamlit as st
import pandas as pd
import os

CACHE_FILE = "weekly_predictions.csv"

st.title("⚽ FPL Points Predictor")

# Check if prediction file exists
if not os.path.exists(CACHE_FILE):
    st.warning("Predictions not found. Please run the pipeline (main.py) first.")
else:
    @st.cache_data
    def load_predictions():
        return pd.read_csv(CACHE_FILE)

    df = load_predictions()

    st.subheader("Predicted Points - All Players")

    # 🔍 Search bar for player name (full table)
    name_search = st.text_input("Search Player Name (for full table below)", "").lower()

    if name_search:
        filtered_df_full = df[df["Player"].str.lower().str.contains(name_search)]
    else:
        filtered_df_full = df

    # Display full (or filtered) table
    st.dataframe(filtered_df_full.sort_values("Predicted Points", ascending=False))

    # Position filter
    st.subheader("🔎 Filter by Position")
    positions = df["Position"].unique()
    selected_position = st.selectbox("Select Position", sorted(positions))
    filtered_df_position = df[df["Position"] == selected_position]

    st.dataframe(filtered_df_position.sort_values("Predicted Points", ascending=False))
