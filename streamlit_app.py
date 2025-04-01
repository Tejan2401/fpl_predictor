import streamlit as st
import pandas as pd
import os
import streamlit as st

st.set_page_config(layout="wide")

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
            # Display editable team with checkbox for captain selection
            team_df_display = team_df[["Player", "Position", "Predicted Points"]].copy()
            team_df_display["Captain?"] = False

            edited_team = st.data_editor(
                team_df_display,
                column_config={
                    "Captain?": st.column_config.CheckboxColumn(
                        label="Captain?",
                        help="Select one player to be captain"
                    )
                },
                use_container_width=True,
                num_rows="fixed"
            )

            captains_selected = edited_team[edited_team["Captain?"] == True]

            if len(captains_selected) == 1:
                edited_team["Adjusted Points"] = edited_team.apply(
                    lambda row: row["Predicted Points"] * 2 if row["Captain?"] else row["Predicted Points"],
                    axis=1
                )
                st.success("✅ Captain selected!")
                st.markdown(f"### 🏆 Total Predicted Points: **{edited_team['Adjusted Points'].sum():.1f}**")

                # Formation layout
                st.markdown("---")
                st.markdown("Pitch View")

                captain_name = captains_selected.iloc[0]["Player"]
                edited_team_full = edited_team.copy()
                edited_team_full["Position"] = team_df.set_index("Player").loc[edited_team["Player"], "Position"].values

                def render_row(players):
                    cols = st.columns(len(players))
                    for col, (_, player) in zip(cols, players.iterrows()):
                        name = player["Player"]
                        is_captain = "⭐" if name == captain_name else ""
                        col.markdown(f"{is_captain} {name}")
                        col.markdown(f"{player['Position']}")
                        col.markdown(f"{player['Predicted Points']:.1f} pts")

                gk = edited_team_full[edited_team_full["Position"] == "GKP"]
                defs = edited_team_full[edited_team_full["Position"] == "DEF"]
                mids = edited_team_full[edited_team_full["Position"] == "MID"]
                fwds = edited_team_full[edited_team_full["Position"] == "FWD"]

                st.markdown("🧤 Goalkeeper")
                render_row(gk)

                st.markdown("🛡️ Defenders")
                render_row(defs)

                st.markdown("🎨 Midfielders")
                render_row(mids)

                st.markdown("🎯 Forwards")
                render_row(fwds)

            elif len(captains_selected) > 1:
                st.error("❌ You can only select one captain.")
            else:
                st.info("☝️ Select a captain to see total points.")

        elif len(selected_players) > 11:
            st.error("You can only select 11 players.")
