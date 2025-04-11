import streamlit as st
import pandas as pd
import os

st.set_page_config(layout="wide")
CACHE_FILE = "weekly_predictions.csv"

st.title("⚽ FPL Points Predictor - GW Selector")

if not os.path.exists(CACHE_FILE):
    st.warning("Predictions not found")
else:
    @st.cache_data
    def load_predictions():
        df = pd.read_csv(CACHE_FILE)

        # Filter for the latest Gameweek
        latest_gw = df["GW"].max()
        df = df[df["GW"] == latest_gw]

        # Collapse double gameweeks by summing predicted points
        df = df.groupby(["Player", "Position"], as_index=False)["Predicted Total Points"].sum()

        # Normalize player names
        df["Player_clean"] = df["Player"].str.strip().str.lower()
        df = df.drop_duplicates(subset=["Player_clean"])

        return df, latest_gw

    df, latest_gw = load_predictions()

    st.subheader(f"Predicted Points for Gameweek {latest_gw}")

    # 🔍 Search bar
    name_search = st.text_input("Search for a player", "").lower()
    filtered_df = df.copy()
    if name_search:
        filtered_df = df[df["Player"].str.lower().str.contains(name_search)]

    filtered_df = filtered_df.sort_values("Predicted Total Points", ascending=False)

    col1, col2 = st.columns([2, 3])

    with col1:
        st.markdown("### 📋 All Players")
        st.dataframe(filtered_df[["Player", "Position", "Predicted Total Points"]])

    with col2:
        st.markdown("### Select Your Starting 11")
        selected_players = st.multiselect(
            "Choose up to 11 players",
            options=filtered_df["Player"].tolist(),
            max_selections=11,
        )

        team_df = df[df["Player"].isin(selected_players)]

        # Count by position
        nb_GK = len(team_df[team_df["Position"] == "GKP"])
        nb_DEF = len(team_df[team_df["Position"] == "DEF"])
        nb_MID = len(team_df[team_df["Position"] == "MID"])
        nb_FWD = len(team_df[team_df["Position"] == "FWD"])

        if selected_players:
            st.markdown(f"Players selected: **{len(selected_players)}** / 11")
            st.markdown(f"GK selected: **{nb_GK}** / 1")
            st.markdown(f"DEF selected: **{nb_DEF}** / min 3")
            st.markdown(f"MID selected: **{nb_MID}** / min 2")
            st.markdown(f"FWD selected: **{nb_FWD}** / min 1")

        if len(selected_players) == 11:
            # Create editable view
            display_df = team_df[["Player", "Position", "Predicted Total Points"]].copy()
            display_df["Captain"] = False

            edited_team = st.data_editor(
                display_df,
                column_config={
                    "Captain": st.column_config.CheckboxColumn(
                        label="Captain",
                        help="Select one player to be captain"
                    )
                },
                use_container_width=True,
                num_rows="fixed"
            )

            # Normalize and merge position info
            edited_team["Player_clean"] = edited_team["Player"].str.strip().str.lower()

            # Merge with df safely
            position_lookup = df[["Player_clean", "Position"]].rename(columns={"Position": "Merged_Position"})
            edited_team_full = edited_team.merge(position_lookup, on="Player_clean", how="left")

            # Assign correct Position field
            edited_team_full["Position"] = edited_team_full["Merged_Position"]

            if "Position" not in edited_team_full.columns or edited_team_full["Position"].isna().any():
                st.error("⚠️ Could not determine player positions after merging.")
                st.write("⚠️ Debug info:", edited_team_full)
            else:
                captains_selected = edited_team_full[edited_team_full["Captain"] == True]

                if len(captains_selected) == 1:
                    captain_name = captains_selected.iloc[0]["Player"]

                    edited_team_full["Adjusted Points"] = edited_team_full.apply(
                        lambda row: row["Predicted Total Points"] * 2 if row["Captain"] else row["Predicted Total Points"],
                        axis=1
                    )

                    st.success("✅ Captain selected!")
                    st.markdown(f"### 🏆 Total Predicted Points: **{edited_team_full['Adjusted Points'].sum():.1f}**")

                    # Render by position
                    def render_row(players):
                        cols = st.columns(len(players))
                        for col, (_, player) in zip(cols, players.iterrows()):
                            is_captain = "⭐" if player["Player"] == captain_name else ""
                            col.markdown(f"{is_captain} {player['Player']}")
                            col.markdown(f"{player['Predicted Total Points']:.1f} pts")

                    gk = edited_team_full[edited_team_full["Position"] == "GKP"]
                    defs = edited_team_full[edited_team_full["Position"] == "DEF"]
                    mids = edited_team_full[edited_team_full["Position"] == "MID"]
                    fwds = edited_team_full[edited_team_full["Position"] == "FWD"]

                    st.markdown("---")
                    st.markdown("🧤 Goalkeeper")
                    render_row(gk)

                    st.markdown("🛡️ Defenders")
                    render_row(defs)

                    st.markdown("🎨 Midfielders")
                    render_row(mids)

                    st.markdown("🎯 Forwards")
                    render_row(fwds)

                elif len(captains_selected) > 1:
                    st.error("🚫 You can only select **one** captain.")
                else:
                    st.info("☑️ Select a captain to see total points.")
        elif len(selected_players) > 11:
            st.error("🚫 You can only select **11** players.")
