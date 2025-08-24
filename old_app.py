import os
from typing import List, Optional

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Optional soccer plotting via mplsoccer (static images)
try:
    from mplsoccer import Pitch
    HAS_MPLSOCCER = True
except Exception:
    HAS_MPLSOCCER = False


# -----------------------------
# Config
# -----------------------------
st.set_page_config(
    page_title="Dinamo București Set-Piece Tactical Report (2024/25)",
    page_icon="⚽",
    layout="wide",
)


# -----------------------------
# Constants
# -----------------------------
SET_PIECE_TYPES_ATTACK = ["corner", "free_kick"]
SET_PIECE_TYPES_ALL = ["corner", "free_kick", "throw_in", "goal_kick"]


# -----------------------------
# Data Loading
# -----------------------------
@st.cache_data(show_spinner=True)
def load_events(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        st.error(f"CSV not found at {csv_path}. Place Dinamo_Bucuresti_2024_2025_events.csv in the app directory.")
        return pd.DataFrame()
    df = pd.read_csv(csv_path, low_memory=False)

    # Basic normalization for convenience
    # Ensure booleans
    for col in [
        "shot.isGoal",
        "shot.onTarget",
        "possession.attack.withShot",
        "possession.attack.withShotOnGoal",
        "possession.attack.withGoal",
    ]:
        if col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.lower().map({"true": True, "false": False})
            df[col] = df[col].fillna(False).astype(bool)

    # Numeric x/y
    for col in ["location.x", "location.y", "pass.endLocation.x", "pass.endLocation.y"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # xG columns
    for col in ["shot.xg", "possession.attack.xg"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df


# -----------------------------
# Helper functions
# -----------------------------
def draw_pitch(fig: Optional[go.Figure] = None,
               pitch_color: str = "#0b6623",
               line_color: str = "white") -> go.Figure:
    """Create a Plotly pitch figure with standard 100x100 coordinate space.
    Assumes x in [0,100], y in [0,100]."""
    if fig is None:
        fig = go.Figure()

    # Pitch background
    fig.add_shape(type="rect", x0=0, y0=0, x1=100, y1=100,
                  line=dict(color=line_color, width=2), fillcolor=pitch_color, layer="below")

    # Halfway line
    fig.add_shape(type="line", x0=50, y0=0, x1=50, y1=100, line=dict(color=line_color, width=2))

    # Penalty boxes and 6-yard boxes (left)
    fig.add_shape(type="rect", x0=0, y0=21.1, x1=16.5, y1=78.9, line=dict(color=line_color, width=2))
    fig.add_shape(type="rect", x0=0, y0=36.8, x1=5.5, y1=63.2, line=dict(color=line_color, width=2))

    # Penalty boxes and 6-yard boxes (right)
    fig.add_shape(type="rect", x0=83.5, y0=21.1, x1=100, y1=78.9, line=dict(color=line_color, width=2))
    fig.add_shape(type="rect", x0=94.5, y0=36.8, x1=100, y1=63.2, line=dict(color=line_color, width=2))

    # Goals (left/right)
    fig.add_shape(type="rect", x0=-1.2, y0=45.2, x1=0, y1=54.8, line=dict(color=line_color, width=2))
    fig.add_shape(type="rect", x0=100, y0=45.2, x1=101.2, y1=54.8, line=dict(color=line_color, width=2))

    # Penalty spots and center spot
    fig.add_shape(type="circle", x0=10.9-0.6, y0=50-0.6, x1=10.9+0.6, y1=50+0.6, line=dict(color=line_color))
    fig.add_shape(type="circle", x0=89.1-0.6, y0=50-0.6, x1=89.1+0.6, y1=50+0.6, line=dict(color=line_color))
    fig.add_shape(type="circle", x0=50-0.6, y0=50-0.6, x1=50+0.6, y1=50+0.6, line=dict(color=line_color))

    # Center circle
    fig.add_shape(type="circle", x0=50-9.15, y0=50-9.15, x1=50+9.15, y1=50+9.15, line=dict(color=line_color, width=2))

    fig.update_xaxes(visible=False, range=[-2, 102])
    fig.update_yaxes(visible=False, range=[-2, 102], scaleanchor="x", scaleratio=1)
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor=pitch_color, paper_bgcolor=pitch_color)
    return fig


def _transform_to_statsbomb_scale(x: pd.Series, y: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Transform 0-100 x,y to StatsBomb 120x80 scale used by mplsoccer."""
    return x * 1.2, y * 0.8


def plot_pitch_mplsoccer_scatter(x: pd.Series, y: pd.Series, c: Optional[pd.Series] = None,
                                 title: Optional[str] = None) -> None:
    """Plot scatter on a soccer pitch using mplsoccer if available."""
    if not HAS_MPLSOCCER:
        st.info("Install mplsoccer for soccer-styled plots: pip install mplsoccer")
        return
    xsb, ysb = _transform_to_statsbomb_scale(x, y)
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#0b6623', line_color='white')
    fig, ax = pitch.draw(figsize=(8, 5))
    colors = None
    if c is not None:
        colors = ["#2ecc71" if bool(val) else "#e74c3c" for val in c]
    pitch.scatter(xsb, ysb, s=60, c=colors, ax=ax, edgecolors='black', linewidth=0.4)
    if title:
        fig.suptitle(title, color='white')
    st.pyplot(fig, use_container_width=True)


def plot_pitch_mplsoccer_heatmap(x: pd.Series, y: pd.Series, title: Optional[str] = None) -> None:
    """Plot heatmap on a soccer pitch using mplsoccer if available."""
    if not HAS_MPLSOCCER:
        st.info("Install mplsoccer for soccer-styled plots: pip install mplsoccer")
        return
    xsb, ysb = _transform_to_statsbomb_scale(x, y)
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#0b6623', line_color='white')
    fig, ax = pitch.draw(figsize=(8, 5))
    try:
        bs = pitch.bin_statistic(xsb, ysb, statistic='count', bins=(30, 20))
        pcm = pitch.heatmap(bs, ax=ax, cmap='YlOrRd')
        fig.colorbar(pcm, ax=ax)
    except Exception:
        pass
    if title:
        fig.suptitle(title, color='white')
    st.pyplot(fig, use_container_width=True)


def identify_set_piece_sequences(df: pd.DataFrame, types: List[str]) -> pd.DataFrame:
    """Return a dataframe of set-piece sequences via possession aggregation.

    Each row represents a possession initiated by a set-piece with summary outcomes.
    """
    if df.empty:
        return pd.DataFrame()

    events = df[df["type.primary"].isin(types)].copy()
    if events.empty:
        return pd.DataFrame()

    cols = [
        "type.primary",
        "possession.id",
        "team.name",
        "possession.duration",
        "possession.attack.withShot",
        "possession.attack.withGoal",
        "possession.attack.xg",
    ]
    subset = events[cols].dropna(subset=["possession.id"]) if "possession.id" in df.columns else events[cols]
    subset = subset.rename(columns={
        "type.primary": "set_piece_type",
        "team.name": "team",
        "possession.attack.withShot": "with_shot",
        "possession.attack.withGoal": "with_goal",
        "possession.attack.xg": "xg_generated",
    })
    subset["with_shot"] = subset["with_shot"].astype(bool)
    subset["with_goal"] = subset["with_goal"].astype(bool)
    subset["xg_generated"] = pd.to_numeric(subset["xg_generated"], errors="coerce").fillna(0.0)
    subset["events_count"] = 1

    # Aggregate at possession level to avoid duplicates if multiple set-piece markers exist in same possession
    agg = subset.groupby(["possession.id", "team", "set_piece_type"], dropna=True).agg(
        events_count=("events_count", "sum"),
        possession_duration=("possession.duration", "max"),
        with_shot=("with_shot", "max"),
        with_goal=("with_goal", "max"),
        xg_generated=("xg_generated", "max"),
    ).reset_index()
    return agg


def filter_offensive_sequences(df: pd.DataFrame) -> pd.DataFrame:
    sequences = identify_set_piece_sequences(df, SET_PIECE_TYPES_ATTACK)
    return sequences[sequences["team"] == "Dinamo Bucureşti"].copy()


def filter_defensive_sequences(df: pd.DataFrame) -> pd.DataFrame:
    sequences = identify_set_piece_sequences(df, SET_PIECE_TYPES_ATTACK)
    return sequences[sequences["team"] != "Dinamo Bucureşti"].copy()


def compute_second_ball_success(df: pd.DataFrame) -> tuple[int, int, float]:
    """Second-ball success: for opponent set-piece possessions, did Dinamo clear/intercept before an opponent shot?"""
    opp_seq = filter_defensive_sequences(df)
    if opp_seq.empty or "possession.id" not in df.columns:
        return (0, 0, 0.0)
    poss_ids = set(opp_seq["possession.id"].unique())
    events = df[df["possession.id"].isin(poss_ids)].copy()
    if events.empty:
        return (0, 0, 0.0)
    success_count = 0
    total = 0
    # Sort as a proxy for event order
    if set(["minute", "second"]).issubset(events.columns):
        events = events.sort_values(["minute", "second"])  # type: ignore
    for pid, group in events.groupby("possession.id"):
        total += 1
        opp_shot_idx = group[(group["team.name"] != "Dinamo Bucureşti") & (group["shot.onTarget"] | group["shot.isGoal"])].index.min()
        dinamo_def_idx = group[(group["team.name"] == "Dinamo Bucureşti") & (group["type.primary"].isin(["clearance", "interception"]))].index.min()
        if pd.isna(opp_shot_idx) and not pd.isna(dinamo_def_idx):
            success_count += 1
        elif not pd.isna(dinamo_def_idx) and not pd.isna(opp_shot_idx) and dinamo_def_idx < opp_shot_idx:
            success_count += 1
    rate = (success_count / total * 100.0) if total > 0 else 0.0
    return (success_count, total, round(rate, 1))


def compute_overview_metrics(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"total_sp": 0, "goals_for": 0, "goals_against": 0, "success_rate": 0.0,
                "type_breakdown": pd.DataFrame(), "gf_ga_by_type": pd.DataFrame()}

    sequences = identify_set_piece_sequences(df, SET_PIECE_TYPES_ATTACK)
    dinamo = sequences[sequences["team"] == "Dinamo Bucureşti"]
    opp = sequences[sequences["team"] != "Dinamo Bucureşti"]

    total_sp = int(dinamo.shape[0])
    goals_for = int(dinamo["with_goal"].sum())
    goals_against = int(opp["with_goal"].sum())
    success_rate = float((dinamo["with_shot"].mean() * 100).round(1)) if total_sp > 0 else 0.0

    type_breakdown = dinamo["set_piece_type"].value_counts().rename_axis("type").reset_index(name="count")

    # Goals for/against by type
    gf = dinamo.groupby("set_piece_type")["with_goal"].sum().rename("goals_for")
    ga = opp.groupby("set_piece_type")["with_goal"].sum().rename("goals_against")
    gf_ga_by_type = pd.concat([gf, ga], axis=1).fillna(0).reset_index().rename(columns={"set_piece_type": "type"})

    return {
        "total_sp": total_sp,
        "goals_for": goals_for,
        "goals_against": goals_against,
        "success_rate": success_rate,
        "type_breakdown": type_breakdown,
        "gf_ga_by_type": gf_ga_by_type,
        "sequences": sequences,
    }


# -----------------------------
# Helpers: Set-Piece DNA catalog and visualization
# -----------------------------
@st.cache_data(show_spinner=False)
def build_set_piece_catalog(df: pd.DataFrame) -> pd.DataFrame:
    """Create a catalog of set-piece possessions with taker, team, start time, and outcome.

    Outcome is computed relative to the initiating team of the set-piece.
    Categories: Goal, Shot on target, Shot off target, No shot.
    """
    if df.empty:
        return pd.DataFrame(columns=[
            "possession.id", "team", "set_piece_type", "taker", "minute", "second", "outcome"
        ])

    if "possession.id" not in df.columns:
        return pd.DataFrame(columns=[
            "possession.id", "team", "set_piece_type", "taker", "minute", "second", "outcome"
        ])

    # Candidates: events that are corners or free kicks
    candidates = df[df["type.primary"].isin(SET_PIECE_TYPES_ATTACK)].copy()
    if candidates.empty:
        return pd.DataFrame(columns=[
            "possession.id", "team", "set_piece_type", "taker", "minute", "second", "outcome"
        ])

    # Normalize ordering
    minute = pd.to_numeric(candidates.get("minute", pd.Series(index=candidates.index, dtype=float)), errors="coerce")
    second = pd.to_numeric(candidates.get("second", pd.Series(index=candidates.index, dtype=float)), errors="coerce")
    candidates["_minute"] = minute.fillna(0)
    candidates["_second"] = second.fillna(0)

    # Pick the first set-piece event per possession as the initiator
    initiators = (
        candidates.dropna(subset=["possession.id"]).sort_values(["possession.id", "_minute", "_second"])  # type: ignore
        .groupby("possession.id", as_index=False)
        .head(1)
    )

    # Compute outcome per possession relative to initiating team
    def _classify_outcome(poss_id: object, team: str) -> str:
        poss_events = df[df["possession.id"] == poss_id].copy()
        if poss_events.empty:
            return "No shot"
        team_events = poss_events[poss_events["team.name"] == team]
        if team_events.empty:
            return "No shot"

        # Identify shot rows
        type_primary = team_events.get("type.primary", pd.Series(index=team_events.index, dtype=object)).astype(str)
        is_shot_row = type_primary.str.contains("shot", case=False, na=False)
        is_goal = team_events.get("shot.isGoal", pd.Series(index=team_events.index, dtype=bool)).fillna(False)
        on_target = team_events.get("shot.onTarget", pd.Series(index=team_events.index, dtype=bool)).fillna(False)
        shot_rows = team_events[is_shot_row | is_goal | on_target]

        if len(shot_rows) == 0:
            return "No shot"
        # Use precomputed flags in case shot columns are absent from dataframe
        if is_goal.loc[shot_rows.index].fillna(False).any():
            return "Goal"
        if on_target.loc[shot_rows.index].fillna(False).any():
            return "Shot on target"
        return "Shot off target"

    catalog = initiators[[
        "possession.id", "team.name", "type.primary", "player.name", "_minute", "_second"
    ]].rename(columns={
        "team.name": "team",
        "type.primary": "set_piece_type",
        "player.name": "taker",
        "_minute": "minute",
        "_second": "second",
    })

    catalog["outcome"] = catalog.apply(lambda r: _classify_outcome(r["possession.id"], r["team"]), axis=1)
    return catalog


def _sort_events(events: pd.DataFrame) -> pd.DataFrame:
    """Sort events chronologically using minute/second fallback to index order."""
    if events.empty:
        return events
    if set(["minute", "second"]).issubset(events.columns):
        mm = pd.to_numeric(events["minute"], errors="coerce").fillna(0)
        ss = pd.to_numeric(events["second"], errors="coerce").fillna(0)
        return events.assign(_m=mm, _s=ss).sort_values(["_m", "_s"]).drop(columns=["_m", "_s"])  # type: ignore
    return events


def get_possession_events(df: pd.DataFrame, possession_id: object) -> pd.DataFrame:
    if df.empty or "possession.id" not in df.columns:
        return pd.DataFrame()
    ev = df[df["possession.id"] == possession_id].copy()
    return _sort_events(ev)


def build_dna_figure(events: pd.DataFrame, highlight_team: Optional[str] = None, animated: bool = False) -> go.Figure:
    """Build a Plotly figure for a possession's event sequence.

    - Passes are lines; highlighted for the selected team
    - Dotted lines approximate same-player movement between their consecutive events
    - Shots are marked and colored by outcome
    - Optional animation to step through events
    """
    fig = draw_pitch()
    if events.empty:
        return fig

    # Prepare numeric coordinates
    sx = pd.to_numeric(events.get("location.x", pd.Series(index=events.index, dtype=float)), errors="coerce")
    sy = pd.to_numeric(events.get("location.y", pd.Series(index=events.index, dtype=float)), errors="coerce")
    ex = pd.to_numeric(events.get("pass.endLocation.x", pd.Series(index=events.index, dtype=float)), errors="coerce")
    ey = pd.to_numeric(events.get("pass.endLocation.y", pd.Series(index=events.index, dtype=float)), errors="coerce")

    # Precompute shot outcome color per row
    is_goal = events.get("shot.isGoal", pd.Series(index=events.index, dtype=bool)).fillna(False)
    on_target = events.get("shot.onTarget", pd.Series(index=events.index, dtype=bool)).fillna(False)
    type_primary = events.get("type.primary", pd.Series(index=events.index, dtype=object)).astype(str)
    is_shot_row = type_primary.str.contains("shot", case=False, na=False) | is_goal | on_target

    def _shot_color(goal: bool, on_t: bool) -> str:
        if goal:
            return "#2ecc71"  # green
        if on_t:
            return "#f1c40f"  # yellow
        return "#e74c3c"      # red

    # Build static traces and optionally frames
    traces_static = []
    frames = []
    previous_location_by_player: dict = {}

    for idx, row in events.iterrows():
        x0, y0 = sx.loc[idx], sy.loc[idx]
        x1, y1 = ex.loc[idx], ey.loc[idx]
        team = str(row.get("team.name", ""))
        player = str(row.get("player.name", ""))
        color = "#1f77b4" if (highlight_team and team == highlight_team) else "#95a5a6"

        step_traces = []

        # Pass lines
        if pd.notna(x0) and pd.notna(y0) and pd.notna(x1) and pd.notna(y1):
            step_traces.append(go.Scatter(
                x=[x0, x1], y=[y0, y1], mode="lines",
                line=dict(color=color, width=3),
                name=f"Pass: {player}",
                showlegend=False,
            ))

        # Same-player movement (approx) as dotted line to new location
        prev_loc = previous_location_by_player.get(player)
        if prev_loc is not None and pd.notna(x0) and pd.notna(y0):
            px0, py0 = prev_loc
            if pd.notna(px0) and pd.notna(py0):
                step_traces.append(go.Scatter(
                    x=[px0, x0], y=[py0, y0], mode="lines",
                    line=dict(color="#7f8c8d", width=2, dash="dot"),
                    name=f"Move: {player}",
                    showlegend=False,
                ))

        # Shot markers
        if bool(is_shot_row.loc[idx]) and pd.notna(x0) and pd.notna(y0):
            sc = _shot_color(bool(is_goal.loc[idx]), bool(on_target.loc[idx]))
            step_traces.append(go.Scatter(
                x=[x0], y=[y0], mode="markers",
                marker=dict(size=12, color=sc, symbol="star"),
                name=f"Shot: {player}",
                showlegend=False,
                text=[player],
                hovertemplate="Shot by %{text}<extra></extra>",
            ))

        # Update last seen location for player
        new_prev = (x1, y1) if (pd.notna(x1) and pd.notna(y1)) else (x0, y0)
        previous_location_by_player[player] = new_prev

        # Accumulate for static
        traces_static.extend(step_traces)

        # Optional animation: cumulative frames
        if animated:
            cumulative = list(traces_static)
            frames.append(go.Frame(data=cumulative, name=str(len(frames) + 1)))

    # Add traces to fig (only when not animating)
    if not animated:
        for t in traces_static:
            fig.add_trace(t)

    if animated and len(frames) > 0:
        fig.frames = frames
        fig.update_layout(
            updatemenus=[{
                "type": "buttons",
                "direction": "left",
                "x": 0.1,
                "y": 1.12,
                "showactive": False,
                "buttons": [
                    {"label": "Play", "method": "animate", "args": [[None], {"frame": {"duration": 700, "redraw": True}, "fromcurrent": True}]},
                    {"label": "Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]}
                ]
            }],
            sliders=[{
                "currentvalue": {"prefix": "Step: "},
                "steps": [{"args": [[f.name], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}], "label": f.name, "method": "animate"} for f in frames]
            }]
        )

    return fig


# -----------------------------
# Page: Overview
# -----------------------------
def page_overview(df: pd.DataFrame) -> None:
    st.header("Overview")
    metrics = compute_overview_metrics(df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Set-Pieces (Corners + Free Kicks)", f"{metrics['total_sp']}")
    c2.metric("Goals Scored from Set-Pieces", f"{metrics['goals_for']}")
    c3.metric("Goals Conceded from Set-Pieces", f"{metrics['goals_against']}")
    c4.metric("Offensive Success Rate (Shot %)", f"{metrics['success_rate']}%")

    st.divider()
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Corners vs Free Kicks (Share)")
        tb = metrics["type_breakdown"].copy()
        tb = tb[tb["type"].isin(["corner", "free_kick"])].sort_values("type")
        if tb.empty:
            st.info("No offensive corners/free kicks found for Dinamo.")
        else:
            fig = px.pie(tb, names="type", values="count", hole=0.5, color="type",
                         color_discrete_map={"corner": "#f94144", "free_kick": "#277da1"})
            fig.update_layout(showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
        with st.expander("What does this show?"):
            st.write("Share of Dinamo's offensive set-pieces by type (corners vs free-kicks).")
        with st.expander("What does this show?"):
            st.write("Share of Dinamo's offensive set-pieces by type (corners vs free-kicks).")

    with col2:
        st.subheader("Goals: For vs Against by Type")
        ga = metrics["gf_ga_by_type"].copy()
        if ga.empty:
            st.info("No goals data available for set-pieces.")
        else:
            ga_melt = ga.melt(id_vars="type", value_vars=["goals_for", "goals_against"],
                              var_name="side", value_name="goals")
            fig2 = px.bar(ga_melt, x="type", y="goals", color="side", barmode="group",
                          color_discrete_map={"goals_for": "#2a9d8f", "goals_against": "#e76f51"})
            st.plotly_chart(fig2, use_container_width=True)
        with st.expander("What does this show?"):
            st.write("Goals scored from Dinamo's set-pieces vs goals conceded from opponent set-pieces, by type.")

    st.subheader("Efficiency by Set-Piece Type")
    ts = metrics.get("type_stats", pd.DataFrame())
    if not ts.empty:
        fig3 = px.bar(ts, x="type", y=["shot_rate", "goal_rate"], barmode="group")
        st.plotly_chart(fig3, use_container_width=True)
        fig4 = px.bar(ts, x="type", y="xg_per_sp", color="type")
        st.plotly_chart(fig4, use_container_width=True)
        with st.expander("What does this show?"):
            st.write("Shot rate, goal rate, and average xG per set-piece — by type.")
        with st.expander("What does this show?"):
            st.write("Goals scored from Dinamo's set-pieces vs goals conceded from opponent set-pieces, by type.")

    st.subheader("Efficiency by Set-Piece Type")
    ts = metrics.get("type_stats", pd.DataFrame())
    if not ts.empty:
        fig3 = px.bar(ts, x="type", y=["shot_rate", "goal_rate"], barmode="group")
        st.plotly_chart(fig3, use_container_width=True)
        fig4 = px.bar(ts, x="type", y="xg_per_sp", color="type")
        st.plotly_chart(fig4, use_container_width=True)
        with st.expander("What does this show?"):
            st.write("Shot rate, goal rate, and average xG per set-piece — by type.")


# -----------------------------
# Page: Set-Piece DNA
# -----------------------------
def page_set_piece_dna(df: pd.DataFrame) -> None:
    st.header("Set-Piece DNA")
    if df.empty:
        st.info("Data not available.")
        return

    catalog = build_set_piece_catalog(df)
    if catalog.empty:
        st.info("No set-piece possessions found.")
        return

    left, right = st.columns([1, 3])

    with left:
        sp_type = st.selectbox("Set-piece type", options=["corner", "free_kick"], index=0)
        teams = ["All"] + sorted(catalog["team"].dropna().unique().tolist())
        team = st.selectbox("Team", options=teams, index=teams.index("All") if "All" in teams else 0)

        filtered = catalog[catalog["set_piece_type"] == sp_type].copy()
        if team != "All":
            filtered = filtered[filtered["team"] == team]

        outcomes = ["All", "Goal", "Shot on target", "Shot off target", "No shot"]
        outcome = st.selectbox("Outcome", options=outcomes, index=0)
        if outcome != "All":
            filtered = filtered[filtered["outcome"] == outcome]

        takers = ["All"] + sorted([t for t in filtered.get("taker", pd.Series()).dropna().unique().tolist()])
        taker = st.selectbox("Taker", options=takers, index=0)
        if taker != "All":
            filtered = filtered[filtered["taker"] == taker]

        if filtered.empty:
            st.info("No sequences match the selected filters.")
            return

        # Selection list
        filtered = filtered.sort_values(["minute", "second"]).reset_index(drop=True)
        options = [
            f"PID {row['possession.id']} — {row['team']} — {row['taker']} — {int(row['minute']):02d}:{int(row['second']):02d} — {row['outcome']}"
            for _, row in filtered.iterrows()
        ]
        sel_label = st.selectbox("Select a set-piece sequence", options=options)
        sel_idx = options.index(sel_label)
        sel_pid = filtered.loc[sel_idx, "possession.id"]

        animated = st.checkbox("Animate sequence", value=False)

    with right:
        ev = get_possession_events(df, sel_pid)
        if ev.empty:
            st.info("No events available for the selected possession.")
            return
        # Show brief meta
        head = filtered.loc[sel_idx]
        st.markdown(f"**Team:** {head['team']} &nbsp;&nbsp; **Taker:** {head['taker']} &nbsp;&nbsp; **Outcome:** {head['outcome']}")
        fig = build_dna_figure(ev, highlight_team=str(head["team"]), animated=animated)
        st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# Page: Offensive Analysis
# -----------------------------
def page_offensive(df: pd.DataFrame) -> None:
    st.header("Offensive Analysis")
    if df.empty:
        st.info("Data not available.")
        return

    sp_type = st.sidebar.selectbox("Select Set-Piece Type", options=["corner", "free_kick"], index=0)
    engine = st.sidebar.selectbox("Pitch Engine", options=["Interactive (Plotly)", "Soccer (mplsoccer)"])

    dinamo_events = df[(df["team.name"] == "Dinamo Bucureşti") & (df["type.primary"] == sp_type)].copy()

    # Primary taker
    st.subheader("Main Taker Analysis")
    if "player.name" in dinamo_events.columns and not dinamo_events.empty and dinamo_events["player.name"].nunique() > 0:
        top_taker = dinamo_events["player.name"].value_counts().idxmax()
    else:
        top_taker = None

    if top_taker:
        st.markdown(f"**Primary {sp_type.replace('_', ' ').title()} Taker:** {top_taker}")
    else:
        st.info("No taker identified for the selected filter.")

    # In-swinger vs Out-swinger proxy
    st.caption("Swing type approximated via pass.angle sign (positive≈out-swinger, negative≈in-swinger). See `https://mplsoccer.readthedocs.io/en/latest/`.")
    angle_series = pd.to_numeric(dinamo_events.get("pass.angle", pd.Series(dtype=float)), errors="coerce")
    swing_df = pd.DataFrame({
        "swing": np.where(angle_series.dropna() >= 0, "Out-swinger", "In-swinger")
    }) if angle_series.notna().any() else pd.DataFrame(columns=["swing"])
    if not swing_df.empty:
        swing_counts = swing_df["swing"].value_counts().reset_index()
        swing_counts.columns = ["swing", "count"]
        fig_swing = px.pie(swing_counts, names="swing", values="count",
                           color="swing", color_discrete_map={"In-swinger": "#8e44ad", "Out-swinger": "#16a085"})
        st.plotly_chart(fig_swing, use_container_width=True)
    else:
        st.info("Insufficient data to infer swing preferences.")

    st.subheader("Target Zone Visualization")
    # Use pass end locations for crosses/deliveries
    endx = pd.to_numeric(dinamo_events.get("pass.endLocation.x", pd.Series(dtype=float)), errors="coerce")
    endy = pd.to_numeric(dinamo_events.get("pass.endLocation.y", pd.Series(dtype=float)), errors="coerce")
    valid_mask = endx.notna() & endy.notna()
    if valid_mask.sum() > 0:
        if engine.startswith("Interactive"):
            fig_hm = draw_pitch()
            fig_hm.add_trace(go.Histogram2dContour(
                x=endx[valid_mask], y=endy[valid_mask],
                contours=dict(coloring="heatmap"),
                colorscale="YlOrRd",
                reversescale=False,
                showscale=True,
                nbinsx=30, nbinsy=30,
            ))
            st.plotly_chart(fig_hm, use_container_width=True)
        else:
            plot_pitch_mplsoccer_heatmap(endx[valid_mask], endy[valid_mask], title="Delivery Heatmap")
    else:
        st.info("No delivery end locations available to plot heatmap.")

    st.subheader("Dangerous Players Analysis")
    # Shots resulting from set-pieces (use possession flags)
    sequences = identify_set_piece_sequences(df, [sp_type])
    dinamo_poss_ids = set(sequences[sequences["team"] == "Dinamo Bucureşti"]["possession.id"].unique())

    shot_events = df[(df["team.name"] == "Dinamo Bucureşti") & (df["shot.onTarget"] | df["shot.isGoal"])]
    if "possession.id" in df.columns and not shot_events.empty:
        sp_shots = shot_events[shot_events["possession.id"].isin(dinamo_poss_ids)]
    else:
        sp_shots = pd.DataFrame()

    if not sp_shots.empty and "player.name" in sp_shots.columns:
        top_shooters = sp_shots["player.name"].value_counts().reset_index()
        top_shooters.columns = ["player", "shots"]
        fig_bar = px.bar(top_shooters.head(15), x="player", y="shots", color="shots", color_continuous_scale="Reds")
        fig_bar.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No shot data available for dangerous players analysis.")

    st.subheader("Shot Map from Set-Pieces")
    shot_x = pd.to_numeric(sp_shots.get("location.x", pd.Series(dtype=float)), errors="coerce")
    shot_y = pd.to_numeric(sp_shots.get("location.y", pd.Series(dtype=float)), errors="coerce")
    if sp_shots.shape[0] > 0 and shot_x.notna().any() and shot_y.notna().any():
        fig_shots = draw_pitch()
        fig_shots.add_trace(go.Scatter(
            x=shot_x, y=shot_y, mode="markers",
            marker=dict(size=10, color=np.where(sp_shots["shot.isGoal"], "#2ecc71", "#e74c3c")),
            text=sp_shots.get("player.name", None), name="Shots"
        ))
        st.plotly_chart(fig_shots, use_container_width=True)
        # # Write in console each shot that resulted in a goal, and all necessary details so that I can search for it in the data
        # for index, row in sp_shots.iterrows():
        #     if row["shot.isGoal"]:
        #         st.write(f"Shot {index} resulted in a goal")
        #         st.write(f"Details: {row}")
        #         st.write(f"Shot ID: {row['id']}")
        #         st.write(f"Shot Location: {row['location.x']}, {row['location.y']}")
        #         st.write(f"Shot Result: {row['shot.isGoal']}")
        #         st.write(f"Shot Possession ID: {row['possession.id']}")
        #         st.write(f"Shot Team: {row['team.name']}")
        #         st.write(f"Shot Player: {row['player.name']}")
    else:
        st.info("No shot location data to plot.")


# -----------------------------
# Page: Defensive Analysis
# -----------------------------
def page_defensive(df: pd.DataFrame) -> None:
    st.header("Defensive Analysis")
    st.info("Zonal, Man-to-Man, and Hybrid are common defensive set-piece systems. Use video/context to annotate observed system here.")

    sequences = identify_set_piece_sequences(df, SET_PIECE_TYPES_ATTACK)
    opp_poss_ids = set(sequences[sequences["team"] != "Dinamo Bucureşti"]["possession.id"].unique())

    # Vulnerability heatmap: shots conceded from opponent set-pieces
    st.subheader("Vulnerability Zones (Shots Conceded)")
    conceded = df[(df["team.name"] != "Dinamo Bucureşti") & (df["shot.onTarget"] | df["shot.isGoal"])].copy()
    if "possession.id" in df.columns and not conceded.empty:
        conceded = conceded[conceded["possession.id"].isin(opp_poss_ids)]
    cx = pd.to_numeric(conceded.get("location.x", pd.Series(dtype=float)), errors="coerce")
    cy = pd.to_numeric(conceded.get("location.y", pd.Series(dtype=float)), errors="coerce")
    valid = cx.notna() & cy.notna()
    if conceded.shape[0] > 0 and valid.sum() > 0:
        fig_conc = draw_pitch()
        fig_conc.add_trace(go.Histogram2dContour(
            x=cx[valid], y=cy[valid], contours=dict(coloring="heatmap"), colorscale="Blues", showscale=True,
            nbinsx=30, nbinsy=30,
        ))
        st.plotly_chart(fig_conc, use_container_width=True)
    else:
        st.info("No conceded shot locations available.")

    # Second Ball Analysis: immediate actions after opponent set-piece cross
    st.subheader("Second Ball Analysis")
    # Proxy: count Dinamo defensive actions within opponent set-piece possessions
    defensive_actions = df[(df["team.name"] == "Dinamo Bucureşti") & (df["type.primary"].isin(["clearance", "interception"]))].copy()
    if "possession.id" in df.columns and not defensive_actions.empty:
        defensive_actions = defensive_actions[defensive_actions["possession.id"].isin(opp_poss_ids)]
    if not defensive_actions.empty and "type.primary" in defensive_actions.columns:
        rates = defensive_actions["type.primary"].value_counts().rename_axis("action").reset_index(name="count")
        fig_second = px.bar(rates, x="action", y="count", color="action")
        st.plotly_chart(fig_second, use_container_width=True)
    else:
        st.info("No defensive actions found within opponent set-piece sequences.")

    # Player Weaknesses: defensive aerial duels lost during set-pieces
    st.subheader("Player Weaknesses: Aerial Duels Lost (Set-Pieces)")
    aerial = df[(df["team.name"] == "Dinamo Bucureşti") & (df["aerialDuel"].notna())].copy()
    if "possession.id" in df.columns and not aerial.empty:
        aerial = aerial[aerial["possession.id"].isin(opp_poss_ids)]
    if not aerial.empty:
        # Identify losses via type.secondary containing 'loss'
        sec = aerial.get("type.secondary", pd.Series(index=aerial.index, dtype=object)).astype(str)
        loss_mask = sec.str.contains("loss", case=False, na=False)
        losses = aerial[loss_mask]
        if not losses.empty and "player.name" in losses.columns:
            table = losses["player.name"].value_counts().reset_index()
            table.columns = ["player", "aerial_duel_losses"]
            st.dataframe(table, use_container_width=True)
        else:
            st.info("No identifiable aerial duel losses during set-pieces.")
    else:
        st.info("No aerial duel data available for set-piece situations.")


# -----------------------------
# Page: Player Profiles
# -----------------------------
def page_player_profiles(df: pd.DataFrame) -> None:
    st.header("Player Profiles")
    dinamo_df = df[df["team.name"] == "Dinamo Bucureşti"].copy()
    players = sorted([p for p in dinamo_df.get("player.name", pd.Series()).dropna().unique()])
    player = st.selectbox("Select player", options=players) if len(players) > 0 else None
    if not player:
        st.info("No players available.")
        return

    left, right = st.columns([1, 3])
    with left:
        st.image("https://placehold.co/300x300?text=Player", caption=player)

    with right:
        st.subheader("Key Set-Piece Stats")
        # Offensive
        player_sp = dinamo_df[(dinamo_df["player.name"] == player) & (dinamo_df["type.primary"].isin(SET_PIECE_TYPES_ATTACK))]
        sp_taken = int(player_sp.shape[0])

        # Successful crosses (proxy: pass.accurate == True within set-piece events)
        accurate = player_sp.get("pass.accurate", pd.Series(dtype=object))
        if accurate.dtype == object:
            accurate = accurate.astype(str).str.lower().map({"true": True, "false": False})
        successful_crosses = int(pd.Series(accurate, dtype="boolean").fillna(False).sum()) if not player_sp.empty else 0

        # Shots and goals from set-piece possessions for this player
        sp_poss_ids = set(player_sp.get("possession.id", pd.Series()).dropna().unique())
        shots_from_sp = df[(df["team.name"] == "Dinamo Bucureşti") & (df["possession.id"].isin(sp_poss_ids)) & (df["shot.onTarget"] | df["shot.isGoal"])].shape[0]
        goals_from_sp = df[(df["team.name"] == "Dinamo Bucureşti") & (df["possession.id"].isin(sp_poss_ids)) & (df["shot.isGoal"])].shape[0]

        # Defensive
        own_box_duels = dinamo_df[(dinamo_df["player.name"] == player) & (dinamo_df["aerialDuel"].notna())]
        sec = own_box_duels.get("type.secondary", pd.Series(index=own_box_duels.index, dtype=object)).astype(str)
        duel_wins = int((~sec.str.contains("loss", case=False, na=False)).sum())
        duel_losses = int((sec.str.contains("loss", case=False, na=False)).sum())
        clearances = int(dinamo_df[(dinamo_df["player.name"] == player) & (dinamo_df["type.primary"] == "clearance")].shape[0])

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Offensive**")
            st.metric("Set-Pieces Taken", f"{sp_taken}")
            st.metric("Successful Crosses", f"{successful_crosses}")
            st.metric("Shots from Set-Pieces", f"{shots_from_sp}")
            st.metric("Goals from Set-Pieces", f"{goals_from_sp}")
        with c2:
            st.markdown("**Defensive**")
            st.metric("Defensive Duels Won", f"{duel_wins}")
            st.metric("Defensive Duels Lost", f"{duel_losses}")
            st.metric("Clearances from Set-Pieces", f"{clearances}")

    st.subheader("Player Shot Map (Set-Pieces)")
    sequences = identify_set_piece_sequences(df, SET_PIECE_TYPES_ATTACK)
    dinamo_poss_ids = set(sequences[sequences["team"] == "Dinamo Bucureşti"]["possession.id"].unique())
    player_shots = df[(df["team.name"] == "Dinamo Bucureşti") & (df["player.name"] == player) & (df["possession.id"].isin(dinamo_poss_ids)) & (df["shot.onTarget"] | df["shot.isGoal"])].copy()
    sx = pd.to_numeric(player_shots.get("location.x", pd.Series(dtype=float)), errors="coerce")
    sy = pd.to_numeric(player_shots.get("location.y", pd.Series(dtype=float)), errors="coerce")
    if player_shots.shape[0] > 0 and sx.notna().any() and sy.notna().any():
        fig_ps = draw_pitch()
        fig_ps.add_trace(go.Scatter(x=sx, y=sy, mode="markers",
                                    marker=dict(size=10, color=np.where(player_shots["shot.isGoal"], "#2ecc71", "#e74c3c")),
                                    name="Shots"))
        st.plotly_chart(fig_ps, use_container_width=True)
    else:
        st.info("No set-piece shots for this player.")

    st.subheader("Player Delivery Map (Corners/Free Kicks)")
    deliveries = dinamo_df[(dinamo_df["player.name"] == player) & (dinamo_df["type.primary"].isin(SET_PIECE_TYPES_ATTACK))].copy()
    ex = pd.to_numeric(deliveries.get("pass.endLocation.x", pd.Series(dtype=float)), errors="coerce")
    ey = pd.to_numeric(deliveries.get("pass.endLocation.y", pd.Series(dtype=float)), errors="coerce")
    if deliveries.shape[0] > 0 and ex.notna().any() and ey.notna().any():
        fig_del = draw_pitch()
        fig_del.add_trace(go.Scatter(x=ex, y=ey, mode="markers", marker=dict(size=8, color="#1f77b4"), name="Deliveries"))
        st.plotly_chart(fig_del, use_container_width=True)
    else:
        st.info("No delivery end locations for this player.")


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    st.title("Dinamo București Set-Piece Tactical Report — 2024/25")
    st.caption("Opponent Analysis Report — Corners and Free Kicks")

    csv_path = os.path.join(os.path.dirname(__file__), "Dinamo_Bucuresti_2024_2025_events.csv")
    df = load_events(csv_path)

    page = st.sidebar.radio(
        "Navigate",
        options=["Overview", "Set-Piece DNA", "Offensive Analysis", "Defensive Analysis", "Player Profiles"],
        index=0,
    )

    if page == "Overview":
        page_overview(df)
    elif page == "Set-Piece DNA":
        page_set_piece_dna(df)
    elif page == "Offensive Analysis":
        page_offensive(df)
    elif page == "Defensive Analysis":
        page_defensive(df)
    elif page == "Player Profiles":
        page_player_profiles(df)


if __name__ == "__main__":
    main()