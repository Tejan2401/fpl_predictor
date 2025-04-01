import streamlit as st
import pandas as pd
import os

CACHE_FILE = "weekly_predictions.csv"

st.title("⚽ FPL Points Predictor")

if not os.path.exists(CACHE_FILE):
    st.warning("Predictions not found. Please run the pipeline (main.py) first.")
else:
    @st.cache_data
    def load_predictions():
        return pd.read_csv(CACHE_FILE)

    df = load_predictions()

    st.subheader("Predicted Points - All Players")

    # Search
    name_search = st.text_input("Search for a player", "").lower()
    if name_search:
        filtered_df_full = df[df["Player"].str.lower().str.contains(name_search)]
    else:
        filtered_df_full = df

    filtered_df_full = filtered_df_full.sort_values("Predicted Points", ascending=False)

    col1, col2 = st.columns([2, 3])

    with col1:
        st.markdown("### 📋 Player Pool")
        st.dataframe(filtered_df_full)

    with col2:
        st.markdown("### ✅ Select Your Starting 11")
        selected_players = st.multiselect(
            "Choose up to 11 players",
            options=filtered_df_full["Player"].tolist(),
            max_selections=11,
        )
        team_df = df[df["Player"].isin(selected_players)]

        if selected_players:
            st.markdown(f"Players selected: **{len(selected_players)}** / 11")

        if len(selected_players) == 11:
            captain = st.selectbox("🎯 Choose Your Captain", selected_players)
            team_df["Captain?"] = team_df["Player"].apply(lambda x: "⭐" if x == captain else "")
            team_df["Adjusted Points"] = team_df.apply(
                lambda row: row["Predicted Points"] * 2 if row["Player"] == captain else row["Predicted Points"],
                axis=1
            )

            # Show team table
            st.markdown("### 📊 Your Team")
            st.dataframe(team_df[["Captain?", "Player", "Position", "Predicted Points", "Adjusted Points"]])
            total = team_df["Adjusted Points"].sum()
            st.markdown(f"### 🏆 Total Predicted Points (with captain): **{total:.1f}**")

            # FORMATION LAYOUT
            st.markdown("---")
            st.markdown("### ⚽ Formation View")

            # Group players
            gk = team_df[team_df["Position"] == "GKP"]
            defs = team_df[team_df["Position"] == "DEF"]
            mids = team_df[team_df["Position"] == "MID"]
            fwds = team_df[team_df["Position"] == "FWD"]

            def render_row(players):
                cols = st.columns(len(players))
                for col, (_, player) in zip(cols, players.iterrows()):
                    name = player["Player"]
                    is_captain = "⭐" if name == captain else ""
                    col.markdown(f"**{is_captain} {name}**")
                    col.markdown(f"*{player['Position']}*")
                    col.markdown(f"{player['Predicted Points']:.1f} pts")

            # Pitch-style layout
            st.markdown("##### 🧤 Goalkeeper")
            render_row(gk)

            st.markdown("##### 🛡️ Defenders")
            render_row(defs)

            st.markdown("##### 🎨 Midfielders")
            render_row(mids)

            st.markdown("##### 🎯 Forwards")
            render_row(fwds)

        elif len(selected_players) > 11:
            st.error("You can only select **11 players**.")
