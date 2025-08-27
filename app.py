import os
from typing import List, Optional, Dict, Tuple
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.figure_factory as ff
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# CONSTANTS
# =============================================================================
SET_PIECE_TYPES_ATTACK = ["corner", "free_kick"]
SET_PIECE_TYPES_ALL = ["corner", "free_kick", "throw_in", "goal_kick"]
DINAMO_COLORS = {
    'primary': '#DC143C',    # Dinamo red
    'secondary': '#FFFFFF',   # White
    'accent': '#FFD700',     # Gold
    'background': '#F8F9FA',
    'text': '#212529'
}

# =============================================================================
# ENHANCED STYLING CONFIGURATION
# =============================================================================
def configure_page_style():
    """Configure professional page styling"""
    st.set_page_config(
        page_title="Dinamo București - Set-Piece Tactical Intelligence",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for professional styling
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #DC143C 0%, #8B0000 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-container {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #DC143C;
        margin-bottom: 1rem;
    }
    .stat-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem;
    }
    .insight-box {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid #28a745;
    }
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffeeba;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid #ffc107;
    }
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid #28a745;
    }
    </style>
    """, unsafe_allow_html=True)

# =============================================================================
# ADVANCED DATA ANALYSIS FUNCTIONS
# =============================================================================

def calculate_advanced_set_piece_metrics(df: pd.DataFrame) -> Dict:
    """Calculate comprehensive set-piece metrics for professional analysis"""
    
    # Filter set pieces
    set_pieces = df[df['type.primary'].isin(SET_PIECE_TYPES_ALL)].copy()
    dinamo_sp = set_pieces[set_pieces['team.name'] == 'Dinamo Bucureşti']
    opponent_sp = set_pieces[set_pieces['team.name'] != 'Dinamo Bucureşti']
    
    # Advanced metrics calculation
    metrics = {
        'total_matches': df['matchId'].nunique(),
        'total_set_pieces': len(set_pieces),
        'dinamo_set_pieces': len(dinamo_sp),
        'opponent_set_pieces': len(opponent_sp),
        'set_pieces_per_match': round(len(dinamo_sp) / df['matchId'].nunique(), 2),
        'set_piece_frequency': {}
    }
    
    # Set piece type breakdown
    for sp_type in SET_PIECE_TYPES_ALL:
        dinamo_count = len(dinamo_sp[dinamo_sp['type.primary'] == sp_type])
        opponent_count = len(opponent_sp[opponent_sp['type.primary'] == sp_type])
        
        metrics['set_piece_frequency'][sp_type] = {
            'dinamo_total': dinamo_count,
            'opponent_total': opponent_count,
            'dinamo_per_match': round(dinamo_count / metrics['total_matches'], 2),
            'opponent_per_match': round(opponent_count / metrics['total_matches'], 2)
        }
    
    # Conversion rates
    attacking_sp = dinamo_sp[dinamo_sp['type.primary'].isin(SET_PIECE_TYPES_ATTACK)]
    if len(attacking_sp) > 0:
        # Ensure numeric columns
        attacking_sp = attacking_sp.copy()
        attacking_sp['possession.attack.withShot'] = pd.to_numeric(attacking_sp['possession.attack.withShot'], errors='coerce').fillna(0)
        attacking_sp['possession.attack.withGoal'] = pd.to_numeric(attacking_sp['possession.attack.withGoal'], errors='coerce').fillna(0)
        attacking_sp['possession.attack.xg'] = pd.to_numeric(attacking_sp['possession.attack.xg'], errors='coerce').fillna(0)
        
        shots_from_sp = attacking_sp['possession.attack.withShot'].sum()
        goals_from_sp = attacking_sp['possession.attack.withGoal'].sum()
        
        metrics['conversion_rates'] = {
            'shot_conversion': round((shots_from_sp / len(attacking_sp)) * 100, 1),
            'goal_conversion': round((goals_from_sp / len(attacking_sp)) * 100, 1),
            'goals_per_shot': round(goals_from_sp / max(shots_from_sp, 1), 2) if shots_from_sp > 0 else 0,
            'total_xg': round(attacking_sp['possession.attack.xg'].sum(), 2),
            'avg_xg_per_sp': round(attacking_sp['possession.attack.xg'].mean(), 3)
        }
    
    # Defensive metrics
    defensive_sp = opponent_sp[opponent_sp['type.primary'].isin(SET_PIECE_TYPES_ATTACK)]
    if len(defensive_sp) > 0:
        # Ensure numeric columns
        defensive_sp = defensive_sp.copy()
        defensive_sp['possession.attack.withShot'] = pd.to_numeric(defensive_sp['possession.attack.withShot'], errors='coerce').fillna(0)
        defensive_sp['possession.attack.withGoal'] = pd.to_numeric(defensive_sp['possession.attack.withGoal'], errors='coerce').fillna(0)
        defensive_sp['possession.attack.xg'] = pd.to_numeric(defensive_sp['possession.attack.xg'], errors='coerce').fillna(0)
        
        shots_conceded = defensive_sp['possession.attack.withShot'].sum()
        goals_conceded = defensive_sp['possession.attack.withGoal'].sum()
        
        metrics['defensive_metrics'] = {
            'shots_conceded_from_sp': shots_conceded,
            'goals_conceded_from_sp': goals_conceded,
            'clean_sheet_rate': round(((len(defensive_sp) - goals_conceded) / len(defensive_sp)) * 100, 1),
            'shots_conceded_rate': round((shots_conceded / len(defensive_sp)) * 100, 1),
            'xg_conceded': round(defensive_sp['possession.attack.xg'].sum(), 2)
        }
    
    return metrics

def analyze_set_piece_timing_patterns(df: pd.DataFrame) -> Dict:
    """Analyze when set pieces occur during matches"""
    set_pieces = df[df['type.primary'].isin(SET_PIECE_TYPES_ATTACK)].copy()
    dinamo_sp = set_pieces[set_pieces['team.name'] == 'Dinamo Bucureşti']
    
    # Time period analysis
    def get_time_period(minute):
        if minute <= 15:
            return '0-15min'
        elif minute <= 30:
            return '16-30min'
        elif minute <= 45:
            return '31-45min'
        elif minute <= 60:
            return '46-60min'
        elif minute <= 75:
            return '61-75min'
        else:
            return '76-90+min'
    
    dinamo_sp['time_period'] = dinamo_sp['minute'].apply(get_time_period)
    
    timing_analysis = {
        'by_period': dinamo_sp['time_period'].value_counts().to_dict(),
        'by_match_period': dinamo_sp['matchPeriod'].value_counts().to_dict(),
        'average_minute': round(dinamo_sp['minute'].mean(), 1),
        'most_productive_period': dinamo_sp[dinamo_sp['possession.attack.withGoal'] == True]['time_period'].mode().iloc[0] if len(dinamo_sp[dinamo_sp['possession.attack.withGoal'] == True]) > 0 else None
    }
    
    return timing_analysis

def analyze_positional_set_piece_data(df: pd.DataFrame) -> Dict:
    """Analyze set piece performance by player positions"""
    set_pieces = df[df['type.primary'].isin(SET_PIECE_TYPES_ATTACK)].copy()
    dinamo_sp = set_pieces[set_pieces['team.name'] == 'Dinamo Bucureşti']
    
    position_analysis = {}
    
    # Analyze by position for different set piece types
    for sp_type in SET_PIECE_TYPES_ATTACK:
        sp_data = dinamo_sp[dinamo_sp['type.primary'] == sp_type]
        if len(sp_data) > 0:
            position_stats = sp_data.groupby('player.position').agg({
                'id': 'count',
                'possession.attack.withShot': 'sum',
                'possession.attack.withGoal': 'sum',
                'possession.attack.xg': 'sum'
            }).reset_index()
            
            position_stats.columns = ['position', 'total_sp', 'shots_generated', 'goals_generated', 'total_xg']
            position_analysis[sp_type] = position_stats.to_dict('records')
    
    return position_analysis

def calculate_set_piece_zones_effectiveness(df: pd.DataFrame) -> Dict:
    """Calculate effectiveness of set pieces from different pitch zones"""
    set_pieces = df[df['type.primary'].isin(SET_PIECE_TYPES_ATTACK)].copy()
    dinamo_sp = set_pieces[set_pieces['team.name'] == 'Dinamo Bucureşti']
    
    # Define zones
    def get_zone(x, y):
        if x <= 33:
            return 'Defensive Third'
        elif x <= 66:
            return 'Middle Third'
        else:
            if y <= 33:
                return 'Attacking Third - Left'
            elif y <= 66:
                return 'Attacking Third - Center'
            else:
                return 'Attacking Third - Right'
    
    dinamo_sp['zone'] = dinamo_sp.apply(lambda row: get_zone(row['location.x'], row['location.y']), axis=1)
    
    # Convert boolean columns to numeric to avoid type errors
    dinamo_sp['possession.attack.withShot'] = pd.to_numeric(dinamo_sp['possession.attack.withShot'], errors='coerce').fillna(0)
    dinamo_sp['possession.attack.withGoal'] = pd.to_numeric(dinamo_sp['possession.attack.withGoal'], errors='coerce').fillna(0)
    dinamo_sp['possession.attack.xg'] = pd.to_numeric(dinamo_sp['possession.attack.xg'], errors='coerce').fillna(0)
    
    zone_analysis = dinamo_sp.groupby(['type.primary', 'zone']).agg({
        'id': 'count',
        'possession.attack.withShot': 'sum',
        'possession.attack.withGoal': 'sum',
        'possession.attack.xg': 'sum'
    }).reset_index()
    
    # Calculate rates with proper numeric handling and division by zero protection
    zone_analysis['shot_rate'] = zone_analysis.apply(
        lambda row: round((row['possession.attack.withShot'] / row['id']) * 100, 1) if row['id'] > 0 else 0.0, axis=1
    )
    zone_analysis['goal_rate'] = zone_analysis.apply(
        lambda row: round((row['possession.attack.withGoal'] / row['id']) * 100, 1) if row['id'] > 0 else 0.0, axis=1
    )
    zone_analysis['avg_xg'] = zone_analysis.apply(
        lambda row: round(row['possession.attack.xg'] / row['id'], 3) if row['id'] > 0 else 0.0, axis=1
    )
    
    # Fill NaN values only in numeric columns
    numeric_cols = zone_analysis.select_dtypes(include=[np.number]).columns
    zone_analysis[numeric_cols] = zone_analysis[numeric_cols].fillna(0)
    
    return zone_analysis.to_dict('records')

# =============================================================================
# HELPER FUNCTIONS FOR SET-PIECE DNA
# =============================================================================
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
        candidates.dropna(subset=["possession.id"]).sort_values(["possession.id", "_minute", "_second"])
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
    
    # Add match context columns that exist in the CSV ('date', 'label')
    catalog = initiators[[
        "possession.id", "team.name", "type.primary", "player.name", "_minute", "_second",
        "date", "label"
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
        return events.assign(_m=mm, _s=ss).sort_values(["_m", "_s"]).drop(columns=["_m", "_s"])
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
    sy = 100 - pd.to_numeric(events.get("location.y", pd.Series(index=events.index, dtype=float)), errors="coerce")
    ex = pd.to_numeric(events.get("pass.endLocation.x", pd.Series(index=events.index, dtype=float)), errors="coerce")
    ey = 100 - pd.to_numeric(events.get("pass.endLocation.y", pd.Series(index=events.index, dtype=float)), errors="coerce")

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

# =============================================================================
# DATA LOADING (Adapted from explore_data.ipynb)
# =============================================================================
@st.cache_data(show_spinner=True)
def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    """
    Load and prepare the dataset for set-piece analysis.
    """
    if not os.path.exists(csv_path):
        st.error(f"CSV not found at {csv_path}. Make sure 'Dinamo_Bucuresti_2024_2025_events.csv' is in the parent directory.")
        return pd.DataFrame()
        
    df = pd.read_csv(csv_path, low_memory=False)
    st.success(f"Dataset loaded successfully: {len(df)} events")
    return df

# =============================================================================
# PITCH CREATION (from explore_data.ipynb)
# =============================================================================
def create_custom_pitch(pitch_color='#2E7D32', line_color='white', stripe_color='#388E3C', stripe_pattern=True):
    """
    Create a professional and realistic football pitch using Plotly,
    scaled to a 0-100 coordinate system. (From explore_data.ipynb)
    """
    fig = go.Figure()

    fig.update_layout(
        xaxis=dict(range=[-5, 105], visible=False),
        yaxis=dict(range=[-5, 105], visible=False),
        plot_bgcolor=pitch_color,
        height=700,
        width=1050,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False
    )

    pitch_shapes = []
    pitch_shapes.append(dict(type='rect', x0=0, y0=0, x1=100, y1=100, line=dict(color=line_color, width=2)))
    pitch_shapes.append(dict(type='line', x0=50, y0=0, x1=50, y1=100, line=dict(color=line_color, width=2)))
    pitch_shapes.append(dict(type='rect', x0=0, y0=21.1, x1=16.5, y1=78.9, line=dict(color=line_color, width=2)))
    pitch_shapes.append(dict(type='rect', x0=83.5, y0=21.1, x1=100, y1=78.9, line=dict(color=line_color, width=2)))
    pitch_shapes.append(dict(type='rect', x0=0, y0=36.8, x1=5.5, y1=63.2, line=dict(color=line_color, width=2)))
    pitch_shapes.append(dict(type='rect', x0=94.5, y0=36.8, x1=100, y1=63.2, line=dict(color=line_color, width=2)))
    pitch_shapes.append(dict(type='circle', x0=40.85, y0=40.85, x1=59.15, y1=59.15, line=dict(color=line_color, width=2)))
    pitch_shapes.append(dict(type='path', path='M 16.5,38.7 C 23.5,43.7, 23.5,56.3, 16.5,61.3', line=dict(color=line_color, width=2)))
    pitch_shapes.append(dict(type='path', path='M 83.5,38.7 C 76.5,43.7, 76.5,56.3, 83.5,61.3', line=dict(color=line_color, width=2)))
    pitch_shapes.append(dict(type='path', path='M 0,1 A 1,1 0 0,1 1,0', line=dict(color=line_color, width=2)))
    pitch_shapes.append(dict(type='path', path='M 0,99 A 1,1 0 0,0 1,100', line=dict(color=line_color, width=2)))
    pitch_shapes.append(dict(type='path', path='M 100,1 A 1,1 0 0,0 99,0', line=dict(color=line_color, width=2)))
    pitch_shapes.append(dict(type='path', path='M 100,99 A 1,1 0 0,1 99,100', line=dict(color=line_color, width=2)))

    if stripe_pattern:
        for i in range(0, 100, 10):
            pitch_shapes.append(dict(
                type='rect', x0=i, y0=0, x1=i+5, y1=100,
                fillcolor=stripe_color, layer='below', line_width=0, opacity=0.3
            ))

    fig.update_layout(shapes=pitch_shapes)
    fig.add_trace(go.Scatter(x=[50], y=[50], mode='markers', marker=dict(color=line_color, size=8), hoverinfo='none'))
    fig.add_trace(go.Scatter(x=[11, 89], y=[50, 50], mode='markers', marker=dict(color=line_color, size=8), hoverinfo='none'))
    fig.add_trace(go.Scatter(x=[-1, -1], y=[45.2, 54.8], mode='lines', line=dict(color=line_color, width=4), hoverinfo='none'))
    fig.add_trace(go.Scatter(x=[101, 101], y=[45.2, 54.8], mode='lines', line=dict(color=line_color, width=4), hoverinfo='none'))

    return fig

# =============================================================================
# ANALYSIS FUNCTIONS (from explore_data.ipynb)
# =============================================================================

# --- Data Preparation Functions ---
def filter_set_pieces(df):
    set_piece_types = ['corner', 'free_kick', 'throw_in', 'goal_kick']
    set_piece_events = df[df['type.primary'].isin(set_piece_types)].copy()
    set_piece_events['set_piece_category'] = set_piece_events['type.primary'].map({
        'corner': 'Corner Kick', 'free_kick': 'Free Kick', 
        'throw_in': 'Throw In', 'goal_kick': 'Goal Kick'
    })
    set_piece_events['is_dinamo'] = set_piece_events['team.name'] == 'Dinamo Bucureşti'
    set_piece_events['context'] = set_piece_events['is_dinamo'].map({
        True: 'Dinamo Attacking', False: 'Dinamo Defending'
    })
    return set_piece_events

def clean_set_piece_data(set_piece_events):
    set_piece_events = set_piece_events.dropna(subset=['location.x', 'location.y', 'player.name'])
    numeric_columns = ['location.x', 'location.y', 'possession.duration', 'possession.attack.xg']
    for col in numeric_columns:
        if col in set_piece_events.columns:
            set_piece_events[col] = pd.to_numeric(set_piece_events[col], errors='coerce')
    return set_piece_events

# --- Offensive Analysis Functions ---
def analyze_offensive_set_pieces(set_piece_events):
    return set_piece_events[
        (set_piece_events['is_dinamo'] == True) & 
        (set_piece_events['type.primary'].isin(['corner', 'free_kick']))
    ].copy()

def identify_main_takers(dinamo_attacking):
    corner_takers = dinamo_attacking[dinamo_attacking['type.primary'] == 'corner']['player.name'].value_counts()
    free_kicks = dinamo_attacking[dinamo_attacking['type.primary'] == 'free_kick'].copy()
    free_kicks['free_kick_type'] = 'Other'
    
    for idx, fk in free_kicks.iterrows():
        x_pos = fk['location.x']
        if x_pos > 66:
            if pd.notna(fk['possession.attack.withShot']) and fk['possession.attack.withShot']:
                free_kicks.loc[idx, 'free_kick_type'] = 'Attacking (Dangerous)'
            else:
                free_kicks.loc[idx, 'free_kick_type'] = 'Attacking (Non-Dangerous)'
        elif x_pos > 33:
            free_kicks.loc[idx, 'free_kick_type'] = 'Middle Third'
        else:
            free_kicks.loc[idx, 'free_kick_type'] = 'Defensive Third'
            
    attacking_dangerous_takers = free_kicks[free_kicks['free_kick_type'] == 'Attacking (Dangerous)']['player.name'].value_counts()
    attacking_non_dangerous_takers = free_kicks[free_kicks['free_kick_type'] == 'Attacking (Non-Dangerous)']['player.name'].value_counts()
    other_free_kick_takers = free_kicks[free_kicks['free_kick_type'].isin(['Middle Third', 'Defensive Third'])]['player.name'].value_counts()
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Top 5 Corner Takers', 'Top 5 Dangerous Attacking Free-Kick Takers',
                       'Top 5 Non-Dangerous Attacking Free-Kick Takers', 'Top 5 Other Free-Kick Takers'),
        specs=[[{"type": "bar"}, {"type": "bar"}], [{"type": "bar"}, {"type": "bar"}]]
    )
    if not corner_takers.empty:
        fig.add_trace(go.Bar(x=corner_takers.head(5).index, y=corner_takers.head(5).values, name='Corner Takers', marker_color='skyblue'), row=1, col=1)
    if not attacking_dangerous_takers.empty:
        fig.add_trace(go.Bar(x=attacking_dangerous_takers.head(5).index, y=attacking_dangerous_takers.head(5).values, name='Dangerous Attacking Free-Kicks', marker_color='red'), row=1, col=2)
    if not attacking_non_dangerous_takers.empty:
        fig.add_trace(go.Bar(x=attacking_non_dangerous_takers.head(5).index, y=attacking_non_dangerous_takers.head(5).values, name='Non-Dangerous Attacking Free-Kicks', marker_color='orange'), row=2, col=1)
    if not other_free_kick_takers.empty:
        fig.add_trace(go.Bar(x=other_free_kick_takers.head(5).index, y=other_free_kick_takers.head(5).values, name='Other Free-Kicks', marker_color='lightcoral'), row=2, col=2)
    
    fig.update_layout(title_text="Comprehensive Set-Piece Takers Analysis", title_x=0.5, height=800, showlegend=False)
    fig.update_xaxes(tickangle=45)
    return fig

def analyze_delivery_zones(dinamo_attacking):
    # Corners
    corners = dinamo_attacking[dinamo_attacking['type.primary'] == 'corner'].copy()
    corners.dropna(subset=['location.x', 'location.y', 'pass.endLocation.x', 'pass.endLocation.y'], inplace=True)
    corners['possession.attack.xg'].fillna(0, inplace=True)
    
    fig_corners = create_custom_pitch()
    if not corners.empty:
        fig_corners.update_layout(title=dict(text="<b>Corner Kick Analysis: Delivery Zones & Danger (xG)</b>", x=0.5, font=dict(size=20, color='white')))
        fig_corners.add_trace(go.Histogram2dContour(x=corners['pass.endLocation.x'], y=100-corners['pass.endLocation.y'], colorscale='Reds', showscale=False, name='Delivery Frequency', contours=dict(coloring='heatmap'), opacity=0.5))
        fig_corners.add_trace(go.Scatter(x=corners['pass.endLocation.x'], y=100-corners['pass.endLocation.y'], mode='markers', marker=dict(color=corners['possession.attack.xg'], colorscale='YlOrRd', size=corners['possession.attack.xg'] * 70 + 10, sizemode='diameter', showscale=True, colorbar=dict(title='xG'), opacity=0.8), name='Delivery Danger (xG)', hoverinfo='text', text=[f"xG: {xg:.3f}" for xg in corners['possession.attack.xg']]))

    # Free Kicks
    attacking_fks = dinamo_attacking[(dinamo_attacking['type.primary'] == 'free_kick') & (dinamo_attacking['location.x'] > 50)].copy()
    attacking_fks.dropna(subset=['location.x', 'location.y', 'pass.endLocation.x', 'pass.endLocation.y'], inplace=True)
    attacking_fks['possession.attack.xg'].fillna(0, inplace=True)

    fig_fks = create_custom_pitch()
    if not attacking_fks.empty:
        fig_fks.update_layout(title=dict(text="<b>Attacking Free Kick Analysis: Delivery Zones & Danger (xG)</b>", x=0.5, font=dict(size=20, color='white')))
        fig_fks.add_trace(go.Histogram2dContour(x=attacking_fks['pass.endLocation.x'], y=100-attacking_fks['pass.endLocation.y'], colorscale='Blues', showscale=False, name='Delivery Frequency', contours=dict(coloring='heatmap'), opacity=0.5))
        fig_fks.add_trace(go.Scatter(x=attacking_fks['pass.endLocation.x'], y=100-attacking_fks['pass.endLocation.y'], mode='markers', marker=dict(color=attacking_fks['possession.attack.xg'], colorscale='Cividis', size=attacking_fks['possession.attack.xg'] * 70 + 10, sizemode='diameter', showscale=True, colorbar=dict(title='xG'), opacity=0.8), name='Delivery Danger (xG)', hoverinfo='text', text=[f"xG: {xg:.3f}" for xg in attacking_fks['possession.attack.xg']]))
        fig_fks.add_trace(go.Scatter(x=attacking_fks['location.x'], y=100-attacking_fks['location.y'], mode='markers', marker=dict(color='cyan', size=8, symbol='x'), name='Free Kick Location'))

    return fig_corners, fig_fks

def detect_attacking_patterns(dinamo_attacking):
    # Ensure numeric columns at the beginning of the function
    dinamo_attacking = dinamo_attacking.copy()
    dinamo_attacking['possession.attack.withShot'] = pd.to_numeric(dinamo_attacking['possession.attack.withShot'], errors='coerce').fillna(0)
    dinamo_attacking['possession.attack.withGoal'] = pd.to_numeric(dinamo_attacking['possession.attack.withGoal'], errors='coerce').fillna(0)
    
    # Corner Strategy
    corners = dinamo_attacking[dinamo_attacking['type.primary'] == 'corner']
    fig_strat = go.Figure()
    if not corners.empty:
        short_corners = 0
        direct_deliveries = 0
        for _, corner in corners.iterrows():
            if pd.notna(corner['pass.recipient.id']) and pd.notna(corner['pass.endLocation.x']) and pd.notna(corner['pass.endLocation.y']):
                distance = np.sqrt((corner['pass.endLocation.x'] - corner['location.x'])**2 + (corner['pass.endLocation.y'] - corner['location.y'])**2)
                if distance < 20:
                    short_corners += 1
                else:
                    direct_deliveries += 1
            else:
                direct_deliveries += 1
        
        fig_strat.add_trace(go.Bar(x=['Short Corners', 'Direct Deliveries'], y=[short_corners, direct_deliveries], marker_color=['lightgreen', 'lightcoral'], text=[short_corners, direct_deliveries], textposition='auto'))
        fig_strat.update_layout(title="Corner Delivery Strategy Analysis", title_x=0.5)

    # Effectiveness
    set_pieces_with_shots = dinamo_attacking[dinamo_attacking['possession.attack.withShot'] == 1]
    set_pieces_with_goals = dinamo_attacking[dinamo_attacking['possession.attack.withGoal'] == 1]
    
    fig_eff = go.Figure()
    categories = ['All Set-Pieces', 'Leading to Shots', 'Leading to Goals']
    values = [len(dinamo_attacking), len(set_pieces_with_shots), len(set_pieces_with_goals)]
    fig_eff.add_trace(go.Bar(x=categories, y=values, marker_color=['lightblue', 'orange', 'red'], text=values, textposition='auto'))
    fig_eff.update_layout(title="Set-Piece Effectiveness Analysis", title_x=0.5)
    
    return fig_strat, fig_eff

def identify_dangerous_players(dinamo_attacking):
    corners = dinamo_attacking[dinamo_attacking['type.primary'] == 'corner']
    attacking_free_kicks = dinamo_attacking[(dinamo_attacking['type.primary'] == 'free_kick') & (dinamo_attacking['location.x'] > 66)]
    other_free_kicks = dinamo_attacking[(dinamo_attacking['type.primary'] == 'free_kick') & (dinamo_attacking['location.x'] <= 66)]
    
    corner_targets = corners[pd.notna(corners['pass.recipient.name'])]['pass.recipient.name'].value_counts()
    attacking_fk_targets = attacking_free_kicks[pd.notna(attacking_free_kicks['pass.recipient.name'])]['pass.recipient.name'].value_counts()
    other_fk_targets = other_free_kicks[pd.notna(other_free_kicks['pass.recipient.name'])]['pass.recipient.name'].value_counts()
    
    fig = make_subplots(rows=1, cols=3, subplot_titles=('Top Corner Targets', 'Top Attacking Free-Kick Targets', 'Top Other Free-Kick Targets'))
    if not corner_targets.empty:
        fig.add_trace(go.Bar(x=corner_targets.head(8).index, y=corner_targets.head(8).values, name='Corner Targets', marker_color='gold'), row=1, col=1)
    if not attacking_fk_targets.empty:
        fig.add_trace(go.Bar(x=attacking_fk_targets.head(8).index, y=attacking_fk_targets.head(8).values, name='Attacking FK Targets', marker_color='red'), row=1, col=2)
    if not other_fk_targets.empty:
        fig.add_trace(go.Bar(x=other_fk_targets.head(8).index, y=other_fk_targets.head(8).values, name='Other FK Targets', marker_color='blue'), row=1, col=3)
        
    fig.update_layout(title_text="Set-Piece Targets by Type", title_x=0.5, height=500, showlegend=False)
    fig.update_xaxes(tickangle=45)
    return fig

# --- Defensive Analysis Functions ---
def analyze_defensive_set_pieces(set_piece_events):
    return set_piece_events[
        (set_piece_events['is_dinamo'] == False) & 
        (set_piece_events['type.primary'].isin(['corner', 'free_kick']))
    ].copy()

def analyze_defensive_vulnerabilities(dinamo_defending, df):
    set_piece_shots_conceded = []
    for _, set_piece in dinamo_defending.iterrows():
        possession_id = set_piece['possession.id']
        if pd.notna(possession_id):
            possession_shots = df[(df['possession.id'] == possession_id) & (df['type.primary'] == 'shot') & (df['team.name'] != 'Dinamo Bucureşti')]
            for _, shot in possession_shots.iterrows():
                set_piece_shots_conceded.append({'x': shot['location.x'], 'y': 100-shot['location.y'], 'is_goal': shot['shot.isGoal'], 'xg': shot['shot.xg'] if pd.notna(shot['shot.xg']) else 0})
    
    fig = create_custom_pitch()
    fig.update_layout(title=dict(text="<b>Dinamo București - Defensive Vulnerabilities from Set-Pieces</b>", x=0.5, font=dict(size=20, color='white')))
    
    if set_piece_shots_conceded:
        shots_df = pd.DataFrame(set_piece_shots_conceded)
        fig.add_trace(go.Histogram2d(x=shots_df['x'], y=shots_df['y'], colorscale='Reds', showscale=False, name='Shot Hotspots', opacity=0.5))
        goals = shots_df[shots_df['is_goal'] == True]
        non_goals = shots_df[shots_df['is_goal'] == False]
        fig.add_trace(go.Scatter(x=non_goals['x'], y=non_goals['y'], mode='markers', marker=dict(color='orange', size=non_goals['xg'] * 50 + 5, sizemode='diameter', opacity=0.8), name='Shots Conceded', hoverinfo='text', text=[f"xG: {xg:.2f}" for xg in non_goals['xg']]))
        fig.add_trace(go.Scatter(x=goals['x'], y=goals['y'], mode='markers', marker=dict(symbol='star', color='red', size=goals['xg'] * 50 + 10, sizemode='diameter', opacity=1.0), name='Goals Conceded', hoverinfo='text', text=[f"GOAL! (xG: {xg:.2f})" for xg in goals['xg']]))
    
    return fig

def analyze_second_ball_reaction(dinamo_defending, df):
    second_ball_analysis = []
    for _, set_piece in dinamo_defending.iterrows():
        possession_id = set_piece['possession.id']
        if pd.notna(possession_id):
            set_piece_time = set_piece['minute'] * 60 + set_piece['second']
            subsequent_events = df[(df['team.name'] == 'Dinamo Bucureşti') & (df['minute'] * 60 + df['second'] > set_piece_time) & (df['minute'] * 60 + df['second'] <= set_piece_time + 30)]
            if len(subsequent_events) > 0:
                time_to_recovery = (subsequent_events.iloc[0]['minute'] * 60 + subsequent_events.iloc[0]['second']) - set_piece_time
                second_ball_analysis.append({'time_to_recovery': time_to_recovery})
    
    fig = go.Figure()
    if second_ball_analysis:
        recovery_df = pd.DataFrame(second_ball_analysis)
        fast_recovery = len(recovery_df[recovery_df['time_to_recovery'] <= 10])
        medium_recovery = len(recovery_df[(recovery_df['time_to_recovery'] > 10) & (recovery_df['time_to_recovery'] <= 20)])
        slow_recovery = len(recovery_df[recovery_df['time_to_recovery'] > 20])
        recovery_counts = [fast_recovery, medium_recovery, slow_recovery]
        recovery_categories = ['Fast (≤10s)', 'Medium (11-20s)', 'Slow (>20s)']
        fig.add_trace(go.Bar(x=recovery_categories, y=recovery_counts, marker_color=['green', 'yellow', 'red'], text=recovery_counts, textposition='auto'))
        fig.update_layout(title="Second Ball Recovery Speed Analysis", title_x=0.5)
        
    return fig

# =============================================================================
# STREAMLIT PAGES
# =============================================================================

# -----------------------------
# Page: Comprehensive Statistics Dashboard
# -----------------------------
def page_comprehensive_stats(df: pd.DataFrame) -> None:
    configure_page_style()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 style="color: white; margin: 0; text-align: center;">
            📊 Comprehensive Set-Piece Intelligence Dashboard
        </h1>
        <p style="color: white; text-align: center; margin-top: 1rem;">
            Advanced tactical analysis of Dinamo București's set-piece performance
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if df.empty:
        st.error("⚠️ No data available for analysis")
        return
    
    # Calculate comprehensive metrics
    with st.spinner('🔄 Calculating advanced metrics...'):
        metrics = calculate_advanced_set_piece_metrics(df)
        timing_analysis = analyze_set_piece_timing_patterns(df)
        positional_data = analyze_positional_set_piece_data(df)
        zone_effectiveness = calculate_set_piece_zones_effectiveness(df)
    
    # Overview Section
    st.markdown("## 📈 Key Performance Indicators")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown("""
        <div class="metric-container">
            <h3 style="color: #DC143C; margin: 0;">Total Matches</h3>
            <h2 style="margin: 0.5rem 0;">{}</h2>
            <small>Season 2024/25</small>
        </div>
        """.format(metrics['total_matches']), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-container">
            <h3 style="color: #DC143C; margin: 0;">Set Pieces per Match</h3>
            <h2 style="margin: 0.5rem 0;">{}</h2>
            <small>Dinamo average</small>
        </div>
        """.format(metrics['set_pieces_per_match']), unsafe_allow_html=True)
    
    with col3:
        if 'conversion_rates' in metrics:
            st.markdown("""
            <div class="metric-container">
                <h3 style="color: #DC143C; margin: 0;">Goal Conversion</h3>
                <h2 style="margin: 0.5rem 0;">{}%</h2>
                <small>From attacking set pieces</small>
            </div>
            """.format(metrics['conversion_rates']['goal_conversion']), unsafe_allow_html=True)
    
    with col4:
        if 'conversion_rates' in metrics:
            st.markdown("""
            <div class="metric-container">
                <h3 style="color: #DC143C; margin: 0;">Expected Goals</h3>
                <h2 style="margin: 0.5rem 0;">{}</h2>
                <small>Total xG from set pieces</small>
            </div>
            """.format(metrics['conversion_rates']['total_xg']), unsafe_allow_html=True)
    
    with col5:
        if 'defensive_metrics' in metrics:
            st.markdown("""
            <div class="metric-container">
                <h3 style="color: #DC143C; margin: 0;">Clean Sheet Rate</h3>
                <h2 style="margin: 0.5rem 0;">{}%</h2>
                <small>Defending set pieces</small>
            </div>
            """.format(metrics['defensive_metrics']['clean_sheet_rate']), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Detailed Analysis Section
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Set Piece Breakdown", 
        "⏰ Timing Analysis", 
        "👥 Positional Analysis", 
        "🗺️ Zone Effectiveness",
        "⚔️ Offensive vs Defensive"
    ])
    
    with tab1:
        st.markdown("### 📊 Set Piece Type Analysis")
        
        # Create detailed breakdown charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Set piece frequency comparison
            sp_data = []
            for sp_type, data in metrics['set_piece_frequency'].items():
                sp_data.append({
                    'Type': sp_type.replace('_', ' ').title(),
                    'Dinamo': data['dinamo_total'],
                    'Opponents': data['opponent_total'],
                    'Dinamo per Match': data['dinamo_per_match'],
                    'Opponents per Match': data['opponent_per_match']
                })
            
            sp_df = pd.DataFrame(sp_data)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Dinamo București',
                x=sp_df['Type'],
                y=sp_df['Dinamo'],
                marker_color=DINAMO_COLORS['primary'],
                text=sp_df['Dinamo'],
                textposition='auto'
            ))
            fig.add_trace(go.Bar(
                name='Opponents',
                x=sp_df['Type'],
                y=sp_df['Opponents'],
                marker_color='#6c757d',
                text=sp_df['Opponents'],
                textposition='auto'
            ))
            
            fig.update_layout(
                title="Set Piece Frequency by Type",
                barmode='group',
                height=400,
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Conversion rates visualization
            if 'conversion_rates' in metrics:
                rates_data = [
                    ['Shot Conversion', metrics['conversion_rates']['shot_conversion']],
                    ['Goal Conversion', metrics['conversion_rates']['goal_conversion']],
                    ['Goals per Shot', metrics['conversion_rates']['goals_per_shot'] * 100]
                ]
                
                fig = go.Figure(go.Bar(
                    x=[item[1] for item in rates_data],
                    y=[item[0] for item in rates_data],
                    orientation='h',
                    marker_color=[DINAMO_COLORS['primary'], DINAMO_COLORS['accent'], '#28a745'],
                    text=[f"{item[1]}%" if 'Conversion' in item[0] else f"{item[1]:.1f}%" for item in rates_data],
                    textposition='auto'
                ))
                
                fig.update_layout(
                    title="Attacking Set Piece Efficiency",
                    height=400,
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Performance insights
        if 'conversion_rates' in metrics:
            cr = metrics['conversion_rates']
            if cr['goal_conversion'] > 15:
                st.markdown("""
                <div class="success-box">
                    <strong>💪 Strength:</strong> Excellent goal conversion rate from set pieces ({}%). 
                    This is above the typical range of 8-12% for professional teams.
                </div>
                """.format(cr['goal_conversion']), unsafe_allow_html=True)
            elif cr['goal_conversion'] < 8:
                st.markdown("""
                <div class="warning-box">
                    <strong>⚠️ Area for Improvement:</strong> Goal conversion rate ({}%) is below average. 
                    Focus on delivery quality and attacking movement patterns.
                </div>
                """.format(cr['goal_conversion']), unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### ⏰ Temporal Set Piece Patterns")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Time period distribution
            periods = list(timing_analysis['by_period'].keys())
            values = list(timing_analysis['by_period'].values())
            
            fig = go.Figure(data=[
                go.Pie(labels=periods, values=values, hole=0.4,
                       marker_colors=px.colors.qualitative.Set3)
            ])
            fig.update_layout(
                title="Set Pieces Distribution by Match Period",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Match half analysis
            if timing_analysis['by_match_period']:
                half_data = timing_analysis['by_match_period']
                
                fig = go.Figure(go.Bar(
                    x=list(half_data.keys()),
                    y=list(half_data.values()),
                    marker_color=[DINAMO_COLORS['primary'], DINAMO_COLORS['accent']],
                    text=list(half_data.values()),
                    textposition='auto'
                ))
                
                fig.update_layout(
                    title="Set Pieces by Match Half",
                    height=400,
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Timing insights
        avg_minute = timing_analysis['average_minute']
        st.markdown(f"""
        <div class="insight-box">
            <strong>📊 Timing Analysis:</strong><br>
            • Average set piece occurs at minute {avg_minute}<br>
            • Most productive period: {timing_analysis.get('most_productive_period', 'N/A')}<br>
            • Pattern suggests {'early pressure' if avg_minute < 45 else 'late opportunities'} tendencies
        </div>
        """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown("### 👥 Position-Based Set Piece Analysis")
        
        for sp_type, pos_data in positional_data.items():
            if pos_data:
                st.markdown(f"#### {sp_type.replace('_', ' ').title()} Analysis")
                
                pos_df = pd.DataFrame(pos_data)
                pos_df = pos_df.sort_values('total_sp', ascending=False)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = go.Figure(go.Bar(
                        x=pos_df['position'],
                        y=pos_df['total_sp'],
                        marker_color=DINAMO_COLORS['primary'],
                        text=pos_df['total_sp'],
                        textposition='auto'
                    ))
                    fig.update_layout(
                        title=f"Total {sp_type.title()} by Position",
                        template="plotly_white",
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    if 'total_xg' in pos_df.columns and pos_df['total_xg'].sum() > 0:
                        fig = go.Figure(go.Bar(
                            x=pos_df['position'],
                            y=pos_df['total_xg'],
                            marker_color=DINAMO_COLORS['accent'],
                            text=pos_df['total_xg'].round(2),
                            textposition='auto'
                        ))
                        fig.update_layout(
                            title=f"xG Generated by Position",
                            template="plotly_white",
                            height=300
                        )
                        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown("### 🗺️ Zone Effectiveness Analysis")
        
        if zone_effectiveness:
            zone_df = pd.DataFrame(zone_effectiveness)
            
            for sp_type in SET_PIECE_TYPES_ATTACK:
                type_data = zone_df[zone_df['type.primary'] == sp_type]
                if not type_data.empty:
                    st.markdown(f"#### {sp_type.replace('_', ' ').title()} Zone Performance")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        fig = go.Figure(go.Bar(
                            x=type_data['zone'],
                            y=type_data['shot_rate'],
                            marker_color='#17a2b8',
                            text=[f"{x}%" for x in type_data['shot_rate']],
                            textposition='auto'
                        ))
                        fig.update_layout(
                            title="Shot Rate by Zone",
                            template="plotly_white",
                            height=300
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        fig = go.Figure(go.Bar(
                            x=type_data['zone'],
                            y=type_data['goal_rate'],
                            marker_color='#28a745',
                            text=[f"{x}%" for x in type_data['goal_rate']],
                            textposition='auto'
                        ))
                        fig.update_layout(
                            title="Goal Rate by Zone",
                            template="plotly_white",
                            height=300
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col3:
                        fig = go.Figure(go.Bar(
                            x=type_data['zone'],
                            y=type_data['avg_xg'],
                            marker_color='#ffc107',
                            text=type_data['avg_xg'],
                            textposition='auto'
                        ))
                        fig.update_layout(
                            title="Average xG by Zone",
                            template="plotly_white",
                            height=300
                        )
                        st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.markdown("### ⚔️ Offensive vs Defensive Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 Attacking Set Pieces")
            if 'conversion_rates' in metrics:
                cr = metrics['conversion_rates']
                
                attack_metrics = [
                    ("Shot Conversion Rate", f"{cr['shot_conversion']}%", "🎯"),
                    ("Goal Conversion Rate", f"{cr['goal_conversion']}%", "⚽"),
                    ("Goals per Shot", f"{cr['goals_per_shot']:.2f}", "🎪"),
                    ("Average xG per Set Piece", f"{cr['avg_xg_per_sp']:.3f}", "📊"),
                    ("Total xG Generated", f"{cr['total_xg']:.2f}", "📈")
                ]
                
                for metric, value, icon in attack_metrics:
                    st.markdown(f"""
                    <div class="stat-card">
                        <h4>{icon} {metric}</h4>
                        <h2 style="color: {DINAMO_COLORS['primary']};">{value}</h2>
                    </div>
                    """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### 🛡️ Defensive Set Pieces")
            if 'defensive_metrics' in metrics:
                dm = metrics['defensive_metrics']
                
                defense_metrics = [
                    ("Clean Sheet Rate", f"{dm['clean_sheet_rate']}%", "🛡️"),
                    ("Shots Conceded Rate", f"{dm['shots_conceded_rate']}%", "⚠️"),
                    ("Goals Conceded", f"{dm['goals_conceded_from_sp']}", "🚫"),
                    ("Shots Conceded", f"{dm['shots_conceded_from_sp']}", "💥"),
                    ("xG Conceded", f"{dm['xg_conceded']:.2f}", "📉")
                ]
                
                for metric, value, icon in defense_metrics:
                    st.markdown(f"""
                    <div class="stat-card">
                        <h4>{icon} {metric}</h4>
                        <h2 style="color: {DINAMO_COLORS['primary']};">{value}</h2>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Strategic Recommendations
    st.markdown("---")
    st.markdown("## 🎯 Strategic Recommendations")
    
    recommendations = generate_tactical_recommendations(metrics, timing_analysis, zone_effectiveness)
    
    for i, (category, recommendation) in enumerate(recommendations.items()):
        if i % 2 == 0:
            box_class = "success-box"
            icon = "💡"
        else:
            box_class = "insight-box"
            icon = "📋"
            
        st.markdown(f"""
        <div class="{box_class}">
            <strong>{icon} {category.title()}:</strong><br>
            {recommendation}
        </div>
        """, unsafe_allow_html=True)

def generate_tactical_recommendations(metrics: Dict, timing_analysis: Dict, zone_effectiveness: List[Dict]) -> Dict:
    """Generate tactical recommendations based on analysis"""
    recommendations = {}
    
    # Offensive recommendations
    if 'conversion_rates' in metrics:
        cr = metrics['conversion_rates']
        if cr['goal_conversion'] < 10:
            recommendations['Attacking Efficiency'] = (
                "Focus on improving set-piece routines and player positioning in the box. "
                f"Current goal conversion of {cr['goal_conversion']}% is below league average. "
                "Consider practicing more varied delivery techniques and timing of runs."
            )
        elif cr['shot_conversion'] < 30:
            recommendations['Shot Creation'] = (
                "Good goal conversion but low shot creation rate. Work on creating more "
                "shooting opportunities from set pieces through better delivery and movement."
            )
    
    # Defensive recommendations
    if 'defensive_metrics' in metrics:
        dm = metrics['defensive_metrics']
        if dm['clean_sheet_rate'] < 70:
            recommendations['Defensive Solidity'] = (
                f"Defensive set-piece performance needs improvement with {dm['clean_sheet_rate']}% clean sheet rate. "
                "Focus on zonal marking, communication, and dealing with second balls."
            )
    
    # Timing recommendations
    if timing_analysis['average_minute'] > 60:
        recommendations['Game Management'] = (
            "Set pieces occur later in matches on average. This could indicate fatigue affecting "
            "defensive discipline or increased attacking urgency. Consider rotation strategies."
        )
    
    return recommendations

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

        # Selection list using the correct 'label' column
        filtered = filtered.sort_values(["date", "minute", "second"]).reset_index(drop=True)
        options = [
            f"{row['label'].split(',')[0]} | {int(row['minute']):02d}' | Taker: {row['taker']} → {row['outcome']}"
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

        # Show detailed meta information parsed from the 'label' and 'date' columns
        head = filtered.loc[sel_idx]
        
        # Parse match information from the 'label' column
        try:
            match_info, final_score = head['label'].split(',')
            home_team, away_team = match_info.split(' - ')
        except ValueError:
            # Fallback if the label format is unexpected
            home_team, away_team, final_score = "Unknown Match", "", ""

        match_date = datetime.strptime(head['date'], '%Y-%m-%d %H:%M:%S').strftime('%d %B %Y')

        st.markdown(f"#### Match: {home_team.strip()} vs. {away_team.strip()}")
        st.markdown(f"**Date:** {match_date}  \n**Final Score:** {final_score.strip()}")
        st.markdown("---")
        
        event_time = f"{int(head['minute']):02d}:{int(head['second']):02d}"
        st.markdown(f"**Event Time:** {event_time}  \n**Set-Piece Taker:** {head['taker']} ({head['team']})  \n**Possession Outcome:** {head['outcome']}")

        # If it was a goal, find and display the scorer
        if head['outcome'] == 'Goal':
            goal_event = ev[(ev['shot.isGoal'] == True) & (ev['team.name'] == head['team'])]
            if not goal_event.empty:
                scorer = goal_event.iloc[0]['player.name']
                st.success(f"⚽ **Goal Scorer:** {scorer}")

        fig = build_dna_figure(ev, highlight_team=str(head["team"]), animated=animated)
        st.plotly_chart(fig, use_container_width=True)

def page_offensive_analysis(df):
    configure_page_style()
    
    st.markdown("""
    <div class="main-header">
        <h1 style="color: white; margin: 0; text-align: center;">
            ⚔️ Offensive Set-Piece Analysis
        </h1>
        <p style="color: white; text-align: center; margin-top: 1rem;">
            Comprehensive tactical breakdown of Dinamo București's attacking set pieces
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if df.empty:
        st.error("⚠️ No data available for analysis")
        return
    
    set_piece_events = filter_set_pieces(df)
    set_piece_events = clean_set_piece_data(set_piece_events)
    dinamo_attacking = analyze_offensive_set_pieces(set_piece_events)
    
    if dinamo_attacking.empty:
        st.warning("No attacking set-pieces for Dinamo to analyze.")
        return
    
    # Quick overview metrics
    total_sp = len(dinamo_attacking)
    # Ensure numeric columns
    dinamo_attacking_copy = dinamo_attacking.copy()
    dinamo_attacking_copy['possession.attack.withShot'] = pd.to_numeric(dinamo_attacking_copy['possession.attack.withShot'], errors='coerce').fillna(0)
    dinamo_attacking_copy['possession.attack.withGoal'] = pd.to_numeric(dinamo_attacking_copy['possession.attack.withGoal'], errors='coerce').fillna(0)
    dinamo_attacking_copy['possession.attack.xg'] = pd.to_numeric(dinamo_attacking_copy['possession.attack.xg'], errors='coerce').fillna(0)
    
    with_shots = dinamo_attacking_copy['possession.attack.withShot'].sum()
    with_goals = dinamo_attacking_copy['possession.attack.withGoal'].sum()
    total_xg = dinamo_attacking_copy['possession.attack.xg'].sum()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <h3 style="color: #DC143C;">Total Set Pieces</h3>
            <h2>{total_sp}</h2>
            <small>Corners & Free Kicks</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        shot_rate = round((with_shots / total_sp) * 100, 1) if total_sp > 0 else 0
        st.markdown(f"""
        <div class="metric-container">
            <h3 style="color: #DC143C;">Shot Rate</h3>
            <h2>{shot_rate}%</h2>
            <small>{with_shots} shots created</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        goal_rate = round((with_goals / total_sp) * 100, 1) if total_sp > 0 else 0
        st.markdown(f"""
        <div class="metric-container">
            <h3 style="color: #DC143C;">Goal Rate</h3>
            <h2>{goal_rate}%</h2>
            <small>{with_goals} goals scored</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_xg = round(total_xg / total_sp, 3) if total_sp > 0 else 0
        st.markdown(f"""
        <div class="metric-container">
            <h3 style="color: #DC143C;">Avg xG per SP</h3>
            <h2>{avg_xg}</h2>
            <small>{round(total_xg, 2)} total xG</small>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Enhanced Analysis Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Set-Piece Takers", 
        "🗺️ Delivery Analysis", 
        "📊 Attack Patterns", 
        "👥 Target Players",
        "🔥 Heat Maps"
    ])
    
    with tab1:
        st.markdown("### 🎯 Set-Piece Specialists Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            with st.spinner('Analyzing takers...'):
                fig_takers = identify_main_takers(dinamo_attacking)
                st.plotly_chart(fig_takers, use_container_width=True)
        
        with col2:
            # Taker efficiency analysis
            st.markdown("#### 📈 Taker Performance")
            
            # Analyze top takers' success rates
            corner_takers = dinamo_attacking[dinamo_attacking['type.primary'] == 'corner']
            fk_takers = dinamo_attacking[dinamo_attacking['type.primary'] == 'free_kick']
            
            if not corner_takers.empty:
                top_corner_taker = corner_takers['player.name'].mode().iloc[0]
                corner_success = corner_takers[corner_takers['player.name'] == top_corner_taker]
                # Ensure numeric conversion
                corner_success = corner_success.copy()
                corner_success['possession.attack.withShot'] = pd.to_numeric(corner_success['possession.attack.withShot'], errors='coerce').fillna(0)
                corner_shot_rate = round((corner_success['possession.attack.withShot'].sum() / len(corner_success)) * 100, 1)
                
                st.markdown(f"""
                <div class="insight-box">
                    <strong>🏆 Top Corner Taker:</strong><br>
                    {top_corner_taker}<br>
                    <strong>Success Rate:</strong> {corner_shot_rate}% shots generated
                </div>
                """, unsafe_allow_html=True)
            
            if not fk_takers.empty:
                top_fk_taker = fk_takers['player.name'].mode().iloc[0]
                fk_success = fk_takers[fk_takers['player.name'] == top_fk_taker]
                # Ensure numeric conversion
                fk_success = fk_success.copy()
                fk_success['possession.attack.withShot'] = pd.to_numeric(fk_success['possession.attack.withShot'], errors='coerce').fillna(0)
                fk_shot_rate = round((fk_success['possession.attack.withShot'].sum() / len(fk_success)) * 100, 1)
                
                st.markdown(f"""
                <div class="insight-box">
                    <strong>🏆 Top Free Kick Taker:</strong><br>
                    {top_fk_taker}<br>
                    <strong>Success Rate:</strong> {fk_shot_rate}% shots generated
                </div>
                """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### 🗺️ Advanced Delivery Zone Analysis")
        
        with st.spinner('Analyzing delivery zones...'):
            fig_corners, fig_fks = analyze_delivery_zones(dinamo_attacking)
            
            st.markdown("#### 🏟️ Corner Kick Delivery Patterns")
            st.plotly_chart(fig_corners, use_container_width=True)
            
            # Corner delivery insights
            corners = dinamo_attacking[dinamo_attacking['type.primary'] == 'corner']
            if not corners.empty:
                corners_with_data = corners.dropna(subset=['pass.endLocation.x', 'pass.endLocation.y'])
                if not corners_with_data.empty:
                    # Analyze delivery zones
                    near_post = len(corners_with_data[corners_with_data['pass.endLocation.y'] < 40])
                    far_post = len(corners_with_data[corners_with_data['pass.endLocation.y'] > 60])
                    central = len(corners_with_data) - near_post - far_post
                    
                    st.markdown(f"""
                    <div class="success-box">
                        <strong>📊 Corner Delivery Preferences:</strong><br>
                        • Near Post: {near_post} deliveries ({round(near_post/len(corners_with_data)*100, 1)}%)<br>
                        • Central Area: {central} deliveries ({round(central/len(corners_with_data)*100, 1)}%)<br>
                        • Far Post: {far_post} deliveries ({round(far_post/len(corners_with_data)*100, 1)}%)
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("#### ⚡ Free Kick Delivery Analysis")
            st.plotly_chart(fig_fks, use_container_width=True)
    
    with tab3:
        st.markdown("### 📊 Attacking Pattern Intelligence")
        
        with st.spinner('Detecting attacking patterns...'):
            fig_strat, fig_eff = detect_attacking_patterns(dinamo_attacking)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(fig_strat, use_container_width=True)
            
            with col2:
                st.plotly_chart(fig_eff, use_container_width=True)
        
        # Enhanced pattern analysis
        st.markdown("#### 🔍 Tactical Patterns Deep Dive")
        
        # Time-based effectiveness
        if 'minute' in dinamo_attacking.columns:
            # Ensure numeric columns
            dinamo_attacking_copy = dinamo_attacking.copy()
            dinamo_attacking_copy['possession.attack.withShot'] = pd.to_numeric(dinamo_attacking_copy['possession.attack.withShot'], errors='coerce').fillna(0)
            dinamo_attacking_copy['possession.attack.withGoal'] = pd.to_numeric(dinamo_attacking_copy['possession.attack.withGoal'], errors='coerce').fillna(0)
            
            effectiveness_by_time = dinamo_attacking_copy.groupby(pd.cut(dinamo_attacking_copy['minute'], 
                                                                  bins=[0, 15, 30, 45, 60, 75, 90], 
                                                                  labels=['0-15', '16-30', '31-45', '46-60', '61-75', '76-90+'])).agg({
                'possession.attack.withShot': 'sum',
                'possession.attack.withGoal': 'sum',
                'id': 'count'
            }).reset_index()
            
            # Handle division by zero and ensure numeric types
            effectiveness_by_time['shot_rate'] = effectiveness_by_time.apply(
                lambda row: round((row['possession.attack.withShot'] / row['id']) * 100, 1) if row['id'] > 0 else 0.0, axis=1
            )
            effectiveness_by_time['goal_rate'] = effectiveness_by_time.apply(
                lambda row: round((row['possession.attack.withGoal'] / row['id']) * 100, 1) if row['id'] > 0 else 0.0, axis=1
            )
            
            # Fill NaN values only in numeric columns
            numeric_cols = effectiveness_by_time.select_dtypes(include=[np.number]).columns
            effectiveness_by_time[numeric_cols] = effectiveness_by_time[numeric_cols].fillna(0)
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig.add_trace(
                go.Bar(x=effectiveness_by_time['minute'], y=effectiveness_by_time['shot_rate'], 
                       name="Shot Rate %", marker_color=DINAMO_COLORS['primary']),
                secondary_y=False
            )
            
            fig.add_trace(
                go.Scatter(x=effectiveness_by_time['minute'], y=effectiveness_by_time['goal_rate'], 
                          name="Goal Rate %", marker_color=DINAMO_COLORS['accent'], mode='lines+markers'),
                secondary_y=True
            )
            
            fig.update_layout(title="Set Piece Effectiveness by Match Period", height=400)
            fig.update_xaxes(title_text="Match Period")
            fig.update_yaxes(title_text="Shot Rate (%)", secondary_y=False)
            fig.update_yaxes(title_text="Goal Rate (%)", secondary_y=True)
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown("### 👥 Target Player Analysis")
        
        with st.spinner('Identifying dangerous players...'):
            fig_players = identify_dangerous_players(dinamo_attacking)
            st.plotly_chart(fig_players, use_container_width=True)
        
        # Enhanced target analysis
        st.markdown("#### 🎯 Target Efficiency Analysis")
        
        # Analyze target success rates
        corners = dinamo_attacking[dinamo_attacking['type.primary'] == 'corner']
        corners_with_targets = corners[corners['pass.recipient.name'].notna()]
        
        if not corners_with_targets.empty:
            # Ensure numeric columns
            corners_with_targets_copy = corners_with_targets.copy()
            corners_with_targets_copy['possession.attack.withShot'] = pd.to_numeric(corners_with_targets_copy['possession.attack.withShot'], errors='coerce').fillna(0)
            corners_with_targets_copy['possession.attack.withGoal'] = pd.to_numeric(corners_with_targets_copy['possession.attack.withGoal'], errors='coerce').fillna(0)
            corners_with_targets_copy['possession.attack.xg'] = pd.to_numeric(corners_with_targets_copy['possession.attack.xg'], errors='coerce').fillna(0)
            
            target_analysis = corners_with_targets_copy.groupby('pass.recipient.name').agg({
                'id': 'count',
                'possession.attack.withShot': 'sum',
                'possession.attack.withGoal': 'sum',
                'possession.attack.xg': 'sum'
            }).reset_index()
            
            # Handle division by zero and ensure numeric types
            target_analysis['shot_rate'] = target_analysis.apply(
                lambda row: round((row['possession.attack.withShot'] / row['id']) * 100, 1) if row['id'] > 0 else 0.0, axis=1
            )
            target_analysis['goal_rate'] = target_analysis.apply(
                lambda row: round((row['possession.attack.withGoal'] / row['id']) * 100, 1) if row['id'] > 0 else 0.0, axis=1
            )
            
            # Fill NaN values only in numeric columns
            numeric_cols = target_analysis.select_dtypes(include=[np.number]).columns
            target_analysis[numeric_cols] = target_analysis[numeric_cols].fillna(0)
            target_analysis = target_analysis.sort_values('id', ascending=False).head(8)
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=target_analysis['shot_rate'],
                y=target_analysis['goal_rate'],
                mode='markers+text',
                marker=dict(
                    size=target_analysis['id'] * 3,
                    color=target_analysis['possession.attack.xg'],
                    colorscale='Reds',
                    showscale=True,
                    colorbar=dict(title="Total xG")
                ),
                text=target_analysis['pass.recipient.name'],
                textposition="top center",
                hovertemplate="<b>%{text}</b><br>Shot Rate: %{x}%<br>Goal Rate: %{y}%<extra></extra>"
            ))
            
            fig.update_layout(
                title="Target Player Efficiency (Bubble size = Targets received)",
                xaxis_title="Shot Rate (%)",
                yaxis_title="Goal Rate (%)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.markdown("### 🔥 Advanced Heat Map Analysis")
        
        # Create enhanced heat maps
        if not dinamo_attacking.empty:
            # Set piece location heat map
            sp_locations = dinamo_attacking.dropna(subset=['location.x', 'location.y'])
            
            if not sp_locations.empty:
                fig = create_custom_pitch()
                fig.update_layout(title="Set Piece Origin Heat Map")
                
                # Add heat map overlay
                fig.add_trace(go.Histogram2d(
                    x=sp_locations['location.x'],
                    y=100-sp_locations['location.y'],
                    colorscale='Reds',
                    showscale=True,
                    opacity=0.7,
                    colorbar=dict(title="Frequency")
                ))
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Performance recommendations
        st.markdown("#### 💡 Tactical Recommendations")
        
        performance_insights = analyze_offensive_performance(dinamo_attacking)
        for insight in performance_insights:
            st.markdown(f"""
            <div class="insight-box">
                {insight}
            </div>
            """, unsafe_allow_html=True)

def analyze_offensive_performance(dinamo_attacking: pd.DataFrame) -> List[str]:
    """Generate tactical insights for offensive set pieces"""
    insights = []
    
    # Ensure numeric columns
    dinamo_attacking = dinamo_attacking.copy()
    dinamo_attacking['possession.attack.withShot'] = pd.to_numeric(dinamo_attacking['possession.attack.withShot'], errors='coerce').fillna(0)
    dinamo_attacking['possession.attack.withGoal'] = pd.to_numeric(dinamo_attacking['possession.attack.withGoal'], errors='coerce').fillna(0)
    
    # Overall effectiveness
    total_sp = len(dinamo_attacking)
    goals_scored = dinamo_attacking['possession.attack.withGoal'].sum()
    shots_created = dinamo_attacking['possession.attack.withShot'].sum()
    
    goal_rate = (goals_scored / total_sp) * 100 if total_sp > 0 else 0
    shot_rate = (shots_created / total_sp) * 100 if total_sp > 0 else 0
    
    if goal_rate > 15:
        insights.append("🔥 <strong>Excellent Goal Conversion:</strong> Your set piece conversion rate is above league average. Continue current tactical approach.")
    elif goal_rate < 8:
        insights.append("⚠️ <strong>Improve Finishing:</strong> Low goal conversion suggests need for better movement in box and delivery precision.")
    
    if shot_rate > 40:
        insights.append("💪 <strong>Strong Shot Creation:</strong> Good at creating chances from set pieces. Focus on improving shot quality.")
    elif shot_rate < 25:
        insights.append("🎯 <strong>Enhance Delivery:</strong> Limited shot creation indicates need for varied delivery options and better timing.")
    
    # Corner vs Free kick analysis
    corners = dinamo_attacking[dinamo_attacking['type.primary'] == 'corner']
    free_kicks = dinamo_attacking[dinamo_attacking['type.primary'] == 'free_kick']
    
    if len(corners) > 0 and len(free_kicks) > 0:
        corner_effectiveness = (corners['possession.attack.withShot'].sum() / len(corners)) * 100
        fk_effectiveness = (free_kicks['possession.attack.withShot'].sum() / len(free_kicks)) * 100
        
        if corner_effectiveness > fk_effectiveness + 10:
            insights.append("📊 <strong>Corner Strength:</strong> Corners are significantly more effective than free kicks. Consider set piece routines transfer.")
        elif fk_effectiveness > corner_effectiveness + 10:
            insights.append("⚡ <strong>Free Kick Excellence:</strong> Free kicks outperform corners. Analyze successful free kick patterns for corner application.")
    
    return insights

def page_defensive_analysis(df):
    configure_page_style()
    
    st.markdown("""
    <div class="main-header">
        <h1 style="color: white; margin: 0; text-align: center;">
            🛡️ Defensive Set-Piece Analysis
        </h1>
        <p style="color: white; text-align: center; margin-top: 1rem;">
            Comprehensive analysis of Dinamo București's defensive set piece performance
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if df.empty:
        st.error("⚠️ No data available for analysis")
        return
    
    set_piece_events = filter_set_pieces(df)
    set_piece_events = clean_set_piece_data(set_piece_events)
    dinamo_defending = analyze_defensive_set_pieces(set_piece_events)
    
    if dinamo_defending.empty:
        st.warning("No defending set-pieces for Dinamo to analyze.")
        return
    
    # Ensure numeric columns are properly converted
    dinamo_defending = dinamo_defending.copy()
    dinamo_defending['possession.attack.withShot'] = pd.to_numeric(dinamo_defending['possession.attack.withShot'], errors='coerce').fillna(0)
    dinamo_defending['possession.attack.withGoal'] = pd.to_numeric(dinamo_defending['possession.attack.withGoal'], errors='coerce').fillna(0)
    dinamo_defending['possession.attack.xg'] = pd.to_numeric(dinamo_defending['possession.attack.xg'], errors='coerce').fillna(0)
    
    # Calculate defensive metrics
    total_def_sp = len(dinamo_defending)
    goals_conceded = len(df[(df['team.name'] != 'Dinamo Bucureşti') & 
                           (df['type.primary'] == 'shot') & 
                           (df['shot.isGoal'] == True) & 
                           (df['possession.id'].isin(dinamo_defending['possession.id']))])
    
    shots_conceded = len(df[(df['team.name'] != 'Dinamo Bucureşti') & 
                           (df['type.primary'] == 'shot') & 
                           (df['possession.id'].isin(dinamo_defending['possession.id']))])
    
    clean_sheets = total_def_sp - goals_conceded
    clean_sheet_rate = round((clean_sheets / total_def_sp) * 100, 1) if total_def_sp > 0 else 0
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <h3 style="color: #DC143C;">Defensive Set Pieces</h3>
            <h2>{total_def_sp}</h2>
            <small>Opponents' corners & free kicks</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <h3 style="color: #DC143C;">Clean Sheet Rate</h3>
            <h2>{clean_sheet_rate}%</h2>
            <small>{clean_sheets} clean defensive displays</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-container">
            <h3 style="color: #DC143C;">Goals Conceded</h3>
            <h2>{goals_conceded}</h2>
            <small>From opponent set pieces</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-container">
            <h3 style="color: #DC143C;">Shots Conceded</h3>
            <h2>{shots_conceded}</h2>
            <small>Total attempts allowed</small>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Enhanced defensive analysis tabs
    tab1, tab3, tab4, tab5 = st.tabs([
        "🚨 Vulnerability Zones", 
        "⚡ Second Ball Analysis", 
        "📊 Defensive Patterns",
        "🎯 Key Insights"
    ])
    
    with tab1:
        st.markdown("### 🚨 Defensive Vulnerability Analysis")
        
        with st.spinner('Analyzing defensive vulnerabilities...'):
            fig_vuln = analyze_defensive_vulnerabilities(dinamo_defending, df)
            st.plotly_chart(fig_vuln, use_container_width=True)
        
        # Enhanced vulnerability analysis
        st.markdown("#### 📍 Danger Zone Breakdown")
        
        # Analyze shots conceded by location
        shots_conceded_data = []
        for _, set_piece in dinamo_defending.iterrows():
            possession_id = set_piece['possession.id']
            if pd.notna(possession_id):
                possession_shots = df[(df['possession.id'] == possession_id) & 
                                    (df['type.primary'] == 'shot') & 
                                    (df['team.name'] != 'Dinamo Bucureşti')]
                for _, shot in possession_shots.iterrows():
                    shots_conceded_data.append({
                        'x': shot['location.x'], 
                        'y': 100-shot['location.y'],
                        'is_goal': shot['shot.isGoal'],
                        'minute': shot['minute'],
                        'set_piece_type': set_piece['type.primary']
                    })
        
        if shots_conceded_data:
            shots_df = pd.DataFrame(shots_conceded_data)
            
            # Zone analysis
            def get_danger_zone(x, y):
                if 83 <= x <= 100:  # Penalty area
                    if 21.1 <= y <= 78.9:
                        if 94.5 <= x <= 100 and 36.8 <= y <= 63.2:
                            return "Six-yard box"
                        else:
                            return "Penalty area"
                    else:
                        return "Wide areas"
                else:
                    return "Outside box"
            
            shots_df['danger_zone'] = shots_df.apply(lambda row: get_danger_zone(row['x'], row['y']), axis=1)
            
            zone_analysis = shots_df.groupby('danger_zone').agg({
                'x': 'count',
                'is_goal': 'sum'
            }).reset_index()
            zone_analysis.columns = ['Zone', 'Shots', 'Goals']
            zone_analysis['Conversion_Rate'] = round((zone_analysis['Goals'] / zone_analysis['Shots']) * 100, 1)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = go.Figure(go.Bar(
                    x=zone_analysis['Zone'],
                    y=zone_analysis['Shots'],
                    marker_color=['#dc3545' if zone == 'Six-yard box' else '#ffc107' if zone == 'Penalty area' else '#28a745' for zone in zone_analysis['Zone']],
                    text=zone_analysis['Shots'],
                    textposition='auto'
                ))
                fig.update_layout(
                    title="Shots Conceded by Zone",
                    template="plotly_white",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = go.Figure(go.Bar(
                    x=zone_analysis['Zone'],
                    y=zone_analysis['Conversion_Rate'],
                    marker_color=['#dc3545' if rate > 20 else '#ffc107' if rate > 10 else '#28a745' for rate in zone_analysis['Conversion_Rate']],
                    text=[f"{rate}%" for rate in zone_analysis['Conversion_Rate']],
                    textposition='auto'
                ))
                fig.update_layout(
                    title="Goal Conversion Rate by Zone",
                    template="plotly_white",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### ⚡ Second Ball Recovery Analysis")
        
        with st.spinner('Analyzing second ball reaction...'):
            fig_recovery = analyze_second_ball_reaction(dinamo_defending, df)
            st.plotly_chart(fig_recovery, use_container_width=True)
        
        # Enhanced second ball analysis
        st.markdown("#### 🔄 Recovery Pattern Analysis")
        
        # Analyze different types of recoveries
        recovery_events = ['clearance', 'interception', 'tackle']
        recovery_analysis = {}
        
        for recovery_type in recovery_events:
            recoveries = df[(df['team.name'] == 'Dinamo Bucureşti') & 
                           (df['type.primary'] == recovery_type)]
            recovery_analysis[recovery_type] = len(recoveries)
        
        if sum(recovery_analysis.values()) > 0:
            fig = go.Figure(go.Pie(
                labels=list(recovery_analysis.keys()),
                values=list(recovery_analysis.values()),
                hole=0.4,
                marker_colors=[DINAMO_COLORS['primary'], DINAMO_COLORS['accent'], '#28a745']
            ))
            fig.update_layout(
                title="Defensive Action Distribution",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown("### 📊 Defensive Pattern Analysis")
        
        # Formation analysis during set pieces
        formations_defending = dinamo_defending['opponentTeam.formation'].value_counts()
        dinamo_formations = dinamo_defending['team.formation'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🏛️ Dinamo's Defensive Formations")
            if not dinamo_formations.empty:
                fig = go.Figure(go.Bar(
                    x=dinamo_formations.index,
                    y=dinamo_formations.values,
                    marker_color=DINAMO_COLORS['primary'],
                    text=dinamo_formations.values,
                    textposition='auto'
                ))
                fig.update_layout(
                    title="Formation Usage When Defending",
                    template="plotly_white",
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### ⚔️ Opposition Formations Faced")
            if not formations_defending.empty:
                fig = go.Figure(go.Bar(
                    x=formations_defending.index,
                    y=formations_defending.values,
                    marker_color='#6c757d',
                    text=formations_defending.values,
                    textposition='auto'
                ))
                fig.update_layout(
                    title="Opposition Formation Distribution",
                    template="plotly_white",
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Time-based defensive performance
        if 'minute' in dinamo_defending.columns:
            # Analyze defensive performance by match period
            dinamo_defending['time_period'] = pd.cut(dinamo_defending['minute'], 
                                                   bins=[0, 15, 30, 45, 60, 75, 90], 
                                                   labels=['0-15', '16-30', '31-45', '46-60', '61-75', '76-90+'])
            
            # Ensure numeric columns before aggregation
            dinamo_defending_copy = dinamo_defending.copy()
            dinamo_defending_copy['possession.attack.withGoal'] = pd.to_numeric(dinamo_defending_copy['possession.attack.withGoal'], errors='coerce').fillna(0)
            dinamo_defending_copy['possession.attack.withShot'] = pd.to_numeric(dinamo_defending_copy['possession.attack.withShot'], errors='coerce').fillna(0)
            
            period_performance = dinamo_defending_copy.groupby('time_period').agg({
                'possession.attack.withGoal': 'sum',
                'possession.attack.withShot': 'sum',
                'id': 'count'
            }).reset_index()
            
            # Handle division by zero and ensure numeric types
            period_performance['goal_rate'] = period_performance.apply(
                lambda row: round((row['possession.attack.withGoal'] / row['id']) * 100, 1) if row['id'] > 0 else 0.0, axis=1
            )
            period_performance['shot_rate'] = period_performance.apply(
                lambda row: round((row['possession.attack.withShot'] / row['id']) * 100, 1) if row['id'] > 0 else 0.0, axis=1
            )
            
            # Fill NaN values only in numeric columns
            numeric_cols = period_performance.select_dtypes(include=[np.number]).columns
            period_performance[numeric_cols] = period_performance[numeric_cols].fillna(0)
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig.add_trace(
                go.Bar(x=period_performance['time_period'], y=period_performance['shot_rate'], 
                       name="Shots Conceded %", marker_color='#ffc107'),
                secondary_y=False
            )
            
            fig.add_trace(
                go.Scatter(x=period_performance['time_period'], y=period_performance['goal_rate'], 
                          name="Goals Conceded %", marker_color='#dc3545', mode='lines+markers'),
                secondary_y=True
            )
            
            fig.update_layout(title="Defensive Performance by Match Period", height=400)
            fig.update_xaxes(title_text="Match Period")
            fig.update_yaxes(title_text="Shots Conceded Rate (%)", secondary_y=False)
            fig.update_yaxes(title_text="Goals Conceded Rate (%)", secondary_y=True)
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.markdown("### 🎯 Key Defensive Insights & Recommendations")
        
        # Generate tactical insights
        defensive_insights = generate_defensive_insights(dinamo_defending, df, clean_sheet_rate, shots_conceded, goals_conceded)
        
        for i, insight in enumerate(defensive_insights):
            if i % 2 == 0:
                box_class = "success-box"
                icon = "💡"
            else:
                box_class = "warning-box"
                icon = "⚠️"
                
            st.markdown(f"""
            <div class="{box_class}">
                <strong>{icon} {insight}</strong>
            </div>
            """, unsafe_allow_html=True)
        
        # Opponent analysis
        st.markdown("#### 🔍 Opposition Set Piece Patterns")
        
        opponent_sp_analysis = dinamo_defending.groupby('team.name').agg({
            'id': 'count',
            'possession.attack.withShot': 'sum',
            'possession.attack.withGoal': 'sum'
        }).reset_index()
        opponent_sp_analysis['shot_rate'] = round((opponent_sp_analysis['possession.attack.withShot'] / opponent_sp_analysis['id']) * 100, 1)
        opponent_sp_analysis['goal_rate'] = round((opponent_sp_analysis['possession.attack.withGoal'] / opponent_sp_analysis['id']) * 100, 1)
        opponent_sp_analysis = opponent_sp_analysis.sort_values('goal_rate', ascending=False)
        
        if not opponent_sp_analysis.empty:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=opponent_sp_analysis['shot_rate'],
                y=opponent_sp_analysis['goal_rate'],
                mode='markers+text',
                marker=dict(
                    size=opponent_sp_analysis['id'] * 2,
                    color=opponent_sp_analysis['goal_rate'],
                    colorscale='Reds',
                    showscale=True,
                    colorbar=dict(title="Goal Rate %")
                ),
                text=opponent_sp_analysis['team.name'],
                textposition="top center",
                hovertemplate="<b>%{text}</b><br>Shot Rate: %{x}%<br>Goal Rate: %{y}%<extra></extra>"
            ))
            
            fig.update_layout(
                title="Opposition Teams: Set Piece Threat Analysis",
                xaxis_title="Shot Rate Against Dinamo (%)",
                yaxis_title="Goal Rate Against Dinamo (%)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)

def generate_defensive_insights(dinamo_defending: pd.DataFrame, df: pd.DataFrame, 
                              clean_sheet_rate: float, shots_conceded: int, goals_conceded: int) -> List[str]:
    """Generate defensive tactical insights"""
    insights = []
    
    total_def_sp = len(dinamo_defending)
    
    # Clean sheet performance
    if clean_sheet_rate > 80:
        insights.append("Excellent Defensive Record: Outstanding clean sheet rate indicates strong set piece defending structure and communication.")
    elif clean_sheet_rate < 60:
        insights.append("Defensive Vulnerability: Low clean sheet rate suggests need for improved marking and positioning during set pieces.")
    
    # Shot prevention
    shot_rate = (shots_conceded / total_def_sp) * 100 if total_def_sp > 0 else 0
    if shot_rate < 30:
        insights.append("Strong Shot Prevention: Good at limiting opponent shooting opportunities from set pieces.")
    elif shot_rate > 50:
        insights.append("Improve Shot Prevention: High rate of shots conceded indicates issues with defensive positioning and pressure.")
    
    # Goal prevention
    if goals_conceded == 0:
        insights.append("Perfect Set Piece Defense: No goals conceded from set pieces demonstrates exceptional defensive organization.")
    elif goals_conceded > 5:
        insights.append("Critical Defensive Issues: High number of goals conceded from set pieces requires immediate tactical attention.")
    
    # Corner vs Free kick analysis
    corner_defending = dinamo_defending[dinamo_defending['type.primary'] == 'corner']
    fk_defending = dinamo_defending[dinamo_defending['type.primary'] == 'free_kick']
    
    if len(corner_defending) > 0 and len(fk_defending) > 0:
        corner_goals = corner_defending['possession.attack.withGoal'].sum()
        fk_goals = fk_defending['possession.attack.withGoal'].sum()
        
        corner_rate = (corner_goals / len(corner_defending)) * 100
        fk_rate = (fk_goals / len(fk_defending)) * 100
        
        if corner_rate > fk_rate + 5:
            insights.append("Corner Weakness: More vulnerable to corners than free kicks. Focus on improving corner defending routines.")
        elif fk_rate > corner_rate + 5:
            insights.append("Free Kick Concern: Struggling more with free kicks than corners. Review wall positioning and goalkeeper coordination.")
    
    return insights

# =============================================================================
# MAIN APP
# =============================================================================
def main():
    # Initialize page configuration first
    configure_page_style()
    
    st.markdown("""
    <div class="main-header">
        <h1 style="color: white; margin: 0; text-align: center;">
            ⚽ Dinamo București - Set-Piece Tactical Intelligence Hub
        </h1>
        <p style="color: white; text-align: center; margin-top: 1rem; font-size: 1.2em;">
            Professional Football Analytics • Season 2024/25 • Advanced Tactical Insights
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Correctly locate the CSV file relative to the app script
    script_dir = os.path.dirname(__file__)
    csv_path = os.path.join(script_dir, "Dinamo_Bucuresti_2024_2025_events.csv")

    # Load data with enhanced error handling
    try:
        df = load_and_prepare_data(csv_path)
        if df.empty:
            st.error("❌ Unable to load data. Please check if the CSV file exists and is properly formatted.")
            return
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return

    # Enhanced sidebar navigation
    with st.sidebar:
        st.markdown("## 🗂️ Navigation")
        st.markdown("---")
        
        # Quick stats in sidebar
        total_matches = df['matchId'].nunique()
        total_events = len(df)
        total_sp = len(df[df['type.primary'].isin(SET_PIECE_TYPES_ALL)])
        
        st.markdown(f"""
        ### 📊 Quick Stats
        - **Matches Analyzed:** {total_matches}
        - **Total Events:** {total_events:,}
        - **Set Pieces:** {total_sp}
        """)
        
        st.markdown("---")
        
        page = st.radio(
            "Select Analysis Page:",
            options=[
                "📊 Comprehensive Statistics",
                "⚔️ Offensive Analysis", 
                "🛡️ Defensive Analysis",
                "🧬 Set-Piece DNA",
            ],
            index=0,
        )
        
        st.markdown("---")
        st.markdown("### 🎯 About This App")
        st.markdown("""
        This advanced analytics platform provides comprehensive 
        tactical insights into Dinamo București's set-piece performance, 
        offering professional-grade analysis for tactical decision making.
        """)
        
        st.markdown("---")
        st.markdown("### 📈 Features")
        st.markdown("""
        • **Real-time Analytics**
        • **Interactive Visualizations** 
        • **Tactical Recommendations**
        • **Performance Benchmarking**
        • **Zone-based Analysis**
        """)

    # Route to appropriate page
    if page == "📊 Comprehensive Statistics":
        page_comprehensive_stats(df)
    elif page == "⚔️ Offensive Analysis":
        page_offensive_analysis(df)
    elif page == "🛡️ Defensive Analysis":
        page_defensive_analysis(df)
    elif page == "🧬 Set-Piece DNA":
        page_set_piece_dna(df)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 10px; margin-top: 2rem;">
        <p style="margin: 0; color: #6c757d;">
            <strong>Dinamo București Set-Piece Intelligence Hub</strong><br>
            Advanced Football Analytics Platform • Season 2024/25<br>
            <small>Professional tactical analysis for data-driven decision making</small>
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
