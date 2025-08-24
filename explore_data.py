#!/usr/bin/env python3
"""
Dinamo București Set-Piece Tactical Analysis
===========================================

This script provides comprehensive analysis of Dinamo București's set-piece strategies
for the 2024/25 season, identifying offensive threats and defensive vulnerabilities.

Author: Tactical Analysis Team
Date: 2024
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set Plotly template for professional appearance
import plotly.io as pio
pio.templates.default = "plotly_white"

# =============================================================================
# CUSTOM PITCH CREATION FOR 0-100 COORDINATE SYSTEM
# =============================================================================

def create_custom_pitch(title="Football Pitch", show_goals=True):
    """
    Create a custom football pitch using Plotly that matches our 0-100 coordinate system
    """
    # Pitch dimensions (0-100 scale)
    pitch_width = 100
    pitch_height = 100
    
    # Create pitch outline
    pitch_outline = go.Scatter(
        x=[0, pitch_width, pitch_width, 0, 0],
        y=[0, 0, pitch_height, pitch_height, 0],
        mode='lines',
        line=dict(color='black', width=3),
        showlegend=False,
        hoverinfo='skip'
    )
    
    # Center line
    center_line = go.Scatter(
        x=[pitch_width/2, pitch_width/2],
        y=[0, pitch_height],
        mode='lines',
        line=dict(color='black', width=2),
        showlegend=False,
        hoverinfo='skip'
    )
    
    # Center circle
    center_circle = go.Scatter(
        x=[pitch_width/2 + 9.15 * np.cos(np.linspace(0, 2*np.pi, 100))],
        y=[pitch_height/2 + 9.15 * np.sin(np.linspace(0, 2*np.pi, 100))],
        mode='lines',
        line=dict(color='black', width=2),
        showlegend=False,
        hoverinfo='skip'
    )
    
    # Penalty areas
    penalty_area_left = go.Scatter(
        x=[0, 16.5, 16.5, 0, 0],
        y=[20, 20, 80, 80, 20],
        mode='lines',
        line=dict(color='black', width=2),
        fill='tonexty',
        fillcolor='rgba(255,255,255,0.1)',
        showlegend=False,
        hoverinfo='skip'
    )
    
    penalty_area_right = go.Scatter(
        x=[pitch_width-16.5, pitch_width, pitch_width, pitch_width-16.5, pitch_width-16.5],
        y=[20, 20, 80, 80, 20],
        mode='lines',
        line=dict(color='black', width=2),
        fill='tonexty',
        fillcolor='rgba(255,255,255,0.1)',
        showlegend=False,
        hoverinfo='skip'
    )
    
    # Goal areas
    goal_area_left = go.Scatter(
        x=[0, 5.5, 5.5, 0, 0],
        y=[35, 35, 65, 65, 35],
        mode='lines',
        line=dict(color='black', width=2),
        fill='tonexty',
        fillcolor='rgba(255,255,255,0.2)',
        showlegend=False,
        hoverinfo='skip'
    )
    
    goal_area_right = go.Scatter(
        x=[pitch_width-5.5, pitch_width, pitch_width, pitch_width-5.5, pitch_width-5.5],
        y=[35, 35, 65, 65, 35],
        mode='lines',
        line=dict(color='black', width=2),
        fill='tonexty',
        fillcolor='rgba(255,255,255,0.2)',
        showlegend=False,
        hoverinfo='skip'
    )
    
    # Penalty spots
    penalty_spot_left = go.Scatter(
        x=[11], y=[pitch_height/2],
        mode='markers',
        marker=dict(color='black', size=8),
        showlegend=False,
        hoverinfo='skip'
    )
    
    penalty_spot_right = go.Scatter(
        x=[pitch_width-11], y=[pitch_height/2],
        mode='markers',
        marker=dict(color='black', size=8),
        showlegend=False,
        hoverinfo='skip'
    )
    
    # Goals
    goals = []
    if show_goals:
        goal_left = go.Scatter(
            x=[-2, -2, 0, 0, -2],
            y=[40, 60, 60, 40, 40],
            mode='lines',
            line=dict(color='white', width=3),
            fill='tonexty',
            fillcolor='white',
            showlegend=False,
            hoverinfo='skip'
        )
        
        goal_right = go.Scatter(
            x=[pitch_width, pitch_width, pitch_width+2, pitch_width+2, pitch_width],
            y=[40, 60, 60, 40, 40],
            mode='lines',
            line=dict(color='white', width=3),
            fill='tonexty',
            fillcolor='white',
            showlegend=False,
            hoverinfo='skip'
        )
        goals = [goal_left, goal_right]
    
    # Zone labels
    near_post_label = go.Scatter(
        x=[8], y=[pitch_height/2],
        mode='text',
        text=['Near Post'],
        textfont=dict(size=12, color='black'),
        showlegend=False,
        hoverinfo='skip'
    )
    
    penalty_spot_label = go.Scatter(
        x=[pitch_width/2], y=[pitch_height/2],
        mode='text',
        text=['Penalty Spot'],
        textfont=dict(size=12, color='black'),
        showlegend=False,
        hoverinfo='skip'
    )
    
    far_post_label = go.Scatter(
        x=[pitch_width-8], y=[pitch_height/2],
        mode='text',
        text=['Far Post'],
        textfont=dict(size=12, color='black'),
        showlegend=False,
        hoverinfo='skip'
    )
    
    # Combine all elements
    pitch_elements = [
        pitch_outline, center_line, center_circle,
        penalty_area_left, penalty_area_right,
        goal_area_left, goal_area_right,
        penalty_spot_left, penalty_spot_right,
        near_post_label, penalty_spot_label, far_post_label
    ] + goals
    
    return pitch_elements

# =============================================================================
# STEP 1: SETUP AND DATA PREPARATION
# =============================================================================

def load_and_prepare_data():
    """
    Load and prepare the dataset for set-piece analysis
    """
    print("=== STEP 1: DATA LOADING AND PREPARATION ===")
    
    try:
        # Load the dataset
        df = pd.read_csv('Dinamo_Bucuresti_2024_2025_events.csv', low_memory=False)
        print(f"✓ Dataset loaded successfully: {len(df)} events")
        
        # Basic dataset info
        print(f"✓ Dataset shape: {df.shape}")
        print(f"✓ Total matches: {df['matchId'].nunique()}")
        print(f"✓ Date range: {df['date'].min()} to {df['date'].max()}")
        
        return df
        
    except FileNotFoundError:
        print("❌ Error: Dataset file not found!")
        return None

def filter_set_pieces(df):
    """
    Filter the dataset for set-piece events only
    """
    print("\n--- Filtering for Set-Piece Events ---")
    
    # Define set-piece event types
    set_piece_types = ['corner', 'free_kick', 'throw_in', 'goal_kick']
    
    # Filter for set-piece events
    set_piece_events = df[df['type.primary'].isin(set_piece_types)].copy()
    
    # Add set-piece classification
    set_piece_events['set_piece_category'] = set_piece_events['type.primary'].map({
        'corner': 'Corner Kick',
        'free_kick': 'Free Kick', 
        'throw_in': 'Throw In',
        'goal_kick': 'Goal Kick'
    })
    
    # Add team context
    set_piece_events['is_dinamo'] = set_piece_events['team.name'] == 'Dinamo Bucureşti'
    set_piece_events['context'] = set_piece_events['is_dinamo'].map({
        True: 'Dinamo Attacking',
        False: 'Dinamo Defending'
    })
    
    print(f"✓ Total set-piece events identified: {len(set_piece_events)}")
    print(f"✓ Set-piece breakdown:")
    for sp_type in set_piece_types:
        count = len(set_piece_events[set_piece_events['type.primary'] == sp_type])
        print(f"  • {sp_type.title()}: {count}")
    
    return set_piece_events

def clean_set_piece_data(set_piece_events):
    """
    Clean and prepare set-piece data for analysis
    """
    print("\n--- Cleaning Set-Piece Data ---")
    
    # Remove events with missing location data
    initial_count = len(set_piece_events)
    set_piece_events = set_piece_events.dropna(subset=['location.x', 'location.y'])
    
    # Remove events with missing player data
    set_piece_events = set_piece_events.dropna(subset=['player.name'])
    
    # Ensure numeric columns are properly typed
    numeric_columns = ['location.x', 'location.y', 'possession.duration', 'possession.attack.xg']
    for col in numeric_columns:
        if col in set_piece_events.columns:
            set_piece_events[col] = pd.to_numeric(set_piece_events[col], errors='coerce')
    
    final_count = len(set_piece_events)
    print(f"✓ Data cleaned: {initial_count - final_count} events removed due to missing data")
    print(f"✓ Final set-piece dataset: {final_count} events")
    
    return set_piece_events

# =============================================================================
# STEP 2: OFFENSIVE SET-PIECE ANALYSIS (DINAMO ATTACKING)
# =============================================================================

def analyze_offensive_set_pieces(set_piece_events):
    """
    Analyze Dinamo's offensive set-piece strategies
    """
    print("\n=== STEP 2: OFFENSIVE SET-PIECE ANALYSIS ===")
    
    # Filter for Dinamo's attacking set-pieces
    dinamo_attacking = set_piece_events[
        (set_piece_events['is_dinamo'] == True) & 
        (set_piece_events['type.primary'].isin(['corner', 'free_kick']))
    ].copy()
    
    print(f"✓ Dinamo attacking set-pieces analyzed: {len(dinamo_attacking)}")
    
    return dinamo_attacking

def identify_main_takers(dinamo_attacking):
    """
    Identify the primary set-piece takers for Dinamo with position-based analysis
    """
    print("\n--- Identifying Main Set-Piece Takers ---")
    
    # Analyze corner takers
    corner_takers = dinamo_attacking[dinamo_attacking['type.primary'] == 'corner']['player.name'].value_counts()
    
    # Analyze free-kick takers with position classification
    free_kicks = dinamo_attacking[dinamo_attacking['type.primary'] == 'free_kick'].copy()
    
    # Classify free-kicks by position and destination
    free_kicks['free_kick_type'] = 'Other'
    
    for idx, fk in free_kicks.iterrows():
        x_pos = fk['location.x']
        
        # Check if it's an attacking free-kick (in attacking third)
        if x_pos > 66:
            # Check if it leads to a shot or goal
            if pd.notna(fk['possession.attack.withShot']) and fk['possession.attack.withShot']:
                free_kicks.loc[idx, 'free_kick_type'] = 'Attacking (Dangerous)'
            else:
                free_kicks.loc[idx, 'free_kick_type'] = 'Attacking (Non-Dangerous)'
        elif x_pos > 33:
            free_kicks.loc[idx, 'free_kick_type'] = 'Middle Third'
        else:
            free_kicks.loc[idx, 'free_kick_type'] = 'Defensive Third'
    
    # Get takers by free-kick type
    attacking_dangerous_takers = free_kicks[free_kicks['free_kick_type'] == 'Attacking (Dangerous)']['player.name'].value_counts()
    attacking_non_dangerous_takers = free_kicks[free_kicks['free_kick_type'] == 'Attacking (Non-Dangerous)']['player.name'].value_counts()
    other_free_kick_takers = free_kicks[free_kicks['free_kick_type'].isin(['Middle Third', 'Defensive Third'])]['player.name'].value_counts()
    
    print("🏃 CORNER TAKERS (Top 5):")
    for i, (player, count) in enumerate(corner_takers.head().items(), 1):
        print(f"  {i}. {player}: {count} corners")
    
    print("\n⚽ ATTACKING FREE-KICK TAKERS (Dangerous - Top 5):")
    for i, (player, count) in enumerate(attacking_dangerous_takers.head().items(), 1):
        print(f"  {i}. {player}: {count} dangerous attacking free-kicks")
    
    print("\n⚽ ATTACKING FREE-KICK TAKERS (Non-Dangerous - Top 5):")
    for i, (player, count) in enumerate(attacking_non_dangerous_takers.head().items(), 1):
        print(f"  {i}. {player}: {count} non-dangerous attacking free-kicks")
    
    print("\n⚽ OTHER FREE-KICK TAKERS (Top 5):")
    for i, (player, count) in enumerate(other_free_kick_takers.head().items(), 1):
        print(f"  {i}. {player}: {count} other free-kicks")
    
    # Create comprehensive visualization using Plotly
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Top 5 Corner Takers', 'Top 5 Dangerous Attacking Free-Kick Takers',
                       'Top 5 Non-Dangerous Attacking Free-Kick Takers', 'Top 5 Other Free-Kick Takers'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    # Corner takers chart
    if len(corner_takers) > 0:
        fig.add_trace(
            go.Bar(
                x=corner_takers.head(5).index,
                y=corner_takers.head(5).values,
                name='Corner Takers',
                marker_color='skyblue',
                marker_line_color='navy',
                marker_line_width=1
            ),
            row=1, col=1
        )
    
    # Dangerous attacking free-kick takers
    if len(attacking_dangerous_takers) > 0:
        fig.add_trace(
            go.Bar(
                x=attacking_dangerous_takers.head(5).index,
                y=attacking_dangerous_takers.head(5).values,
                name='Dangerous Attacking Free-Kicks',
                marker_color='red',
                marker_line_color='darkred',
                marker_line_width=1
            ),
            row=1, col=2
        )
    
    # Non-dangerous attacking free-kick takers
    if len(attacking_non_dangerous_takers) > 0:
        fig.add_trace(
            go.Bar(
                x=attacking_non_dangerous_takers.head(5).index,
                y=attacking_non_dangerous_takers.head(5).values,
                name='Non-Dangerous Attacking Free-Kicks',
                marker_color='orange',
                marker_line_color='darkorange',
                marker_line_width=1
            ),
            row=2, col=1
        )
    
    # Other free-kick takers
    if len(other_free_kick_takers) > 0:
        fig.add_trace(
            go.Bar(
                x=other_free_kick_takers.head(5).index,
                y=other_free_kick_takers.head(5).values,
                name='Other Free-Kicks',
                marker_color='lightcoral',
                marker_line_color='darkred',
                marker_line_width=1
            ),
            row=2, col=2
        )
    
    # Update layout
    fig.update_layout(
        title_text="Comprehensive Set-Piece Takers Analysis",
        title_x=0.5,
        height=800,
        showlegend=False
    )
    
    # Update x-axis labels for better readability
    fig.update_xaxes(tickangle=45)
    
    # Save and show
    fig.write_html('set_piece_takers_comprehensive.html')
    fig.show()
    
    return corner_takers, attacking_dangerous_takers, attacking_non_dangerous_takers, other_free_kick_takers

def analyze_delivery_zones(dinamo_attacking):
    """
    Analyze the target zones and delivery areas for set-pieces using professional pitch visualization
    """
    print("\n--- Analyzing Delivery Zones ---")
    
    # Analyze corner delivery zones with heatmap
    corners = dinamo_attacking[dinamo_attacking['type.primary'] == 'corner']
    
    if len(corners) > 0:
        # Get delivery end locations for corners
        corner_deliveries = []
        for _, corner in corners.iterrows():
            if pd.notna(corner['pass.endLocation.x']) and pd.notna(corner['pass.endLocation.y']):
                corner_deliveries.append([corner['pass.endLocation.x'], corner['pass.endLocation.y']])
        
        if corner_deliveries:
            corner_deliveries = np.array(corner_deliveries)
            
            # Create custom pitch
            pitch_elements = create_custom_pitch(title="Corner Delivery Zones Analysis")
            
            # Create heatmap using Plotly
            fig = go.Figure()
            
            # Add pitch elements
            for element in pitch_elements:
                fig.add_trace(element)
            
            # Add heatmap for corner deliveries
            if len(corner_deliveries) > 1:
                # Create 2D histogram for heatmap effect
                x_bins = np.linspace(0, 100, 25)
                y_bins = np.linspace(0, 100, 25)
                
                H, xedges, yedges = np.histogram2d(
                    corner_deliveries[:, 0], corner_deliveries[:, 1], 
                    bins=[x_bins, y_bins]
                )
                
                # Create heatmap
                heatmap = go.Heatmap(
                    z=H.T,
                    x=xedges[:-1],
                    y=yedges[:-1],
                    colorscale='Reds',
                    opacity=0.7,
                    showscale=True,
                    name='Delivery Density'
                )
                fig.add_trace(heatmap)
            
            # Add individual delivery points
            delivery_points = go.Scatter(
                x=corner_deliveries[:, 0],
                y=corner_deliveries[:, 1],
                mode='markers',
                marker=dict(
                    color='red',
                    size=6,
                    opacity=0.6
                ),
                name='Corner Deliveries',
                text=[f'Delivery {i+1}' for i in range(len(corner_deliveries))],
                hoverinfo='text'
            )
            fig.add_trace(delivery_points)
            
            # Zone analysis
            near_post = len(corner_deliveries[corner_deliveries[:, 0] < 20])
            penalty_area = len(corner_deliveries[(corner_deliveries[:, 0] >= 20) & (corner_deliveries[:, 0] <= 80)])
            far_post = len(corner_deliveries[corner_deliveries[:, 0] > 80])
            
            # Add statistics text
            stats_text = go.Scatter(
                x=[50], y=[95],
                mode='text',
                text=[f'Corner Analysis: {len(corners)} corners analyzed<br>Near Post: {near_post} | Penalty Area: {penalty_area} | Far Post: {far_post}'],
                textfont=dict(size=14, color='black'),
                showlegend=False,
                hoverinfo='skip'
            )
            fig.add_trace(stats_text)
            
            # Update layout
            fig.update_layout(
                title="Corner Delivery Zones Analysis (Heatmap)",
                title_x=0.5,
                xaxis=dict(range=[-5, 105], showgrid=False, zeroline=False),
                yaxis=dict(range=[-5, 105], showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1),
                height=700,
                showlegend=True
            )
            
            # Save and show
            fig.write_html('corner_delivery_zones_heatmap.html')
            fig.show()
            
            print(f"✓ Corner delivery zones analyzed: {len(corners)} corners")
            print(f"  • Near post deliveries: {near_post}")
            print(f"  • Penalty area deliveries: {penalty_area}")
            print(f"  • Far post deliveries: {far_post}")
    
    # Analyze attacking free-kick delivery zones (only dangerous ones)
    attacking_free_kicks = dinamo_attacking[
        (dinamo_attacking['type.primary'] == 'free_kick') & 
        (dinamo_attacking['location.x'] > 66)
    ]
    
    if len(attacking_free_kicks) > 0:
        # Get delivery end locations for attacking free-kicks
        fk_deliveries = []
        for _, fk in attacking_free_kicks.iterrows():
            if pd.notna(fk['pass.endLocation.x']) and pd.notna(fk['pass.endLocation.y']):
                fk_deliveries.append([fk['pass.endLocation.x'], fk['pass.endLocation.y']])
        
        if fk_deliveries:
            fk_deliveries = np.array(fk_deliveries)
            
            # Create custom pitch
            pitch_elements = create_custom_pitch(title="Attacking Free-Kick Delivery Zones Analysis")
            
            # Create heatmap using Plotly
            fig = go.Figure()
            
            # Add pitch elements
            for element in pitch_elements:
                fig.add_trace(element)
            
            # Add heatmap for free-kick deliveries
            if len(fk_deliveries) > 1:
                # Create 2D histogram for heatmap effect
                x_bins = np.linspace(0, 100, 25)
                y_bins = np.linspace(0, 100, 25)
                
                H, xedges, yedges = np.histogram2d(
                    fk_deliveries[:, 0], fk_deliveries[:, 1], 
                    bins=[x_bins, y_bins]
                )
                
                # Create heatmap
                heatmap = go.Heatmap(
                    z=H.T,
                    x=xedges[:-1],
                    y=yedges[:-1],
                    colorscale='Blues',
                    opacity=0.7,
                    showscale=True,
                    name='Delivery Density'
                )
                fig.add_trace(heatmap)
            
            # Add individual delivery points
            delivery_points = go.Scatter(
                x=fk_deliveries[:, 0],
                y=fk_deliveries[:, 1],
                mode='markers',
                marker=dict(
                    color='blue',
                    size=6,
                    opacity=0.6
                ),
                name='Attacking Free-Kick Deliveries',
                text=[f'Delivery {i+1}' for i in range(len(fk_deliveries))],
                hoverinfo='text'
            )
            fig.add_trace(delivery_points)
            
            # Zone analysis
            near_post = len(fk_deliveries[fk_deliveries[:, 0] < 20])
            penalty_area = len(fk_deliveries[(fk_deliveries[:, 0] >= 20) & (fk_deliveries[:, 0] <= 80)])
            far_post = len(fk_deliveries[fk_deliveries[:, 0] > 80])
            
            # Add statistics text
            stats_text = go.Scatter(
                x=[50], y=[95],
                mode='text',
                text=[f'Attacking Free-Kick Analysis: {len(attacking_free_kicks)} free-kicks analyzed<br>Near Post: {near_post} | Penalty Area: {penalty_area} | Far Post: {far_post}'],
                textfont=dict(size=14, color='black'),
                showlegend=False,
                hoverinfo='skip'
            )
            fig.add_trace(stats_text)
            
            # Update layout
            fig.update_layout(
                title="Attacking Free-Kick Delivery Zones Analysis (Heatmap)",
                title_x=0.5,
                xaxis=dict(range=[-5, 105], showgrid=False, zeroline=False),
                yaxis=dict(range=[-5, 105], showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1),
                height=700,
                showlegend=True
            )
            
            # Save and show
            fig.write_html('attacking_free_kick_delivery_zones_heatmap.html')
            fig.show()
            
            print(f"✓ Attacking free-kick delivery zones analyzed: {len(attacking_free_kicks)} free-kicks")
            print(f"  • Near post deliveries: {near_post}")
            print(f"  • Penalty area deliveries: {penalty_area}")
            print(f"  • Far post deliveries: {far_post}")

def detect_attacking_patterns(dinamo_attacking):
    """
    Detect common attacking routines and patterns
    """
    print("\n--- Detecting Attacking Patterns ---")
    
    # Analyze short corners vs direct deliveries
    corners = dinamo_attacking[dinamo_attacking['type.primary'] == 'corner']
    
    if len(corners) > 0:
        # Identify short corners (delivery to nearby teammate)
        short_corners = 0
        direct_deliveries = 0
        
        for _, corner in corners.iterrows():
            # Check if there's a pass recipient nearby (within 20 units)
            if pd.notna(corner['pass.recipient.id']):
                recipient_x = corner['pass.endLocation.x']
                recipient_y = corner['pass.endLocation.y']
                corner_x = corner['location.x']
                corner_y = corner['location.y']
                
                if pd.notna(recipient_x) and pd.notna(recipient_y):
                    distance = np.sqrt((recipient_x - corner_x)**2 + (recipient_y - corner_y)**2)
                    if distance < 20:
                        short_corners += 1
                    else:
                        direct_deliveries += 1
                else:
                    direct_deliveries += 1
            else:
                direct_deliveries += 1
        
        # Create visualization using Plotly
        fig = go.Figure()
        
        categories = ['Short Corners', 'Direct Deliveries']
        values = [short_corners, direct_deliveries]
        colors = ['lightgreen', 'lightcoral']
        
        fig.add_trace(go.Bar(
            x=categories,
            y=values,
            marker_color=colors,
            marker_line_color='black',
            marker_line_width=2,
            text=values,
            textposition='auto',
            textfont=dict(size=14, color='black')
        ))
        
        fig.update_layout(
            title="Corner Delivery Strategy Analysis",
            title_x=0.5,
            xaxis_title="Strategy Type",
            yaxis_title="Number of Corners",
            height=500,
            showlegend=False
        )
        
        # Save and show
        fig.write_html('corner_strategy_analysis.html')
        fig.show()
        
        print(f"✓ Corner strategy analysis completed:")
        print(f"  • Short corners: {short_corners} ({short_corners/len(corners)*100:.1f}%)")
        print(f"  • Direct deliveries: {direct_deliveries} ({direct_deliveries/len(corners)*100:.1f}%)")
    
    # Analyze set-piece sequences
    print("\n--- Set-Piece Sequence Analysis ---")
    
    # Count set-pieces leading to shots
    set_pieces_with_shots = dinamo_attacking[
        dinamo_attacking['possession.attack.withShot'] == True
    ]
    
    set_pieces_with_goals = dinamo_attacking[
        dinamo_attacking['possession.attack.withGoal'] == True
    ]
    
    print(f"✓ Set-pieces leading to shots: {len(set_pieces_with_shots)} ({len(set_pieces_with_shots)/len(dinamo_attacking)*100:.1f}%)")
    print(f"✓ Set-pieces leading to goals: {len(set_pieces_with_goals)} ({len(set_pieces_with_goals)/len(dinamo_attacking)*100:.1f}%)")
    
    # Create visualization for set-piece effectiveness using Plotly
    fig = go.Figure()
    
    categories = ['All Set-Pieces', 'Leading to Shots', 'Leading to Goals']
    values = [len(dinamo_attacking), len(set_pieces_with_shots), len(set_pieces_with_goals)]
    colors = ['lightblue', 'orange', 'red']
    
    # Calculate percentages for text labels
    percentages = []
    for i, value in enumerate(values):
        if i == 0:
            percentages.append('100%')
        else:
            percentages.append(f'{value/len(dinamo_attacking)*100:.1f}%')
    
    fig.add_trace(go.Bar(
        x=categories,
        y=values,
        marker_color=colors,
        marker_line_color='black',
        marker_line_width=2,
        text=[f'{v}<br>({p})' for v, p in zip(values, percentages)],
        textposition='auto',
        textfont=dict(size=12, color='black')
    ))
    
    fig.update_layout(
        title="Set-Piece Effectiveness Analysis",
        title_x=0.5,
        xaxis_title="Event Type",
        yaxis_title="Number of Events",
        height=500,
        showlegend=False
    )
    
    # Save and show
    fig.write_html('set_piece_effectiveness.html')
    fig.show()

def identify_dangerous_players(dinamo_attacking):
    """
    Identify the most dangerous players in set-piece situations, split by type
    """
    print("\n--- Identifying Dangerous Players by Set-Piece Type ---")
    
    # Split set-pieces by type
    corners = dinamo_attacking[dinamo_attacking['type.primary'] == 'corner']
    attacking_free_kicks = dinamo_attacking[
        (dinamo_attacking['type.primary'] == 'free_kick') & 
        (dinamo_attacking['location.x'] > 66)
    ]
    other_free_kicks = dinamo_attacking[
        (dinamo_attacking['type.primary'] == 'free_kick') & 
        (dinamo_attacking['location.x'] <= 66)
    ]
    
    # Analyze targets by set-piece type
    corner_targets = corners[pd.notna(corners['pass.recipient.name'])]['pass.recipient.name'].value_counts()
    attacking_fk_targets = attacking_free_kicks[pd.notna(attacking_free_kicks['pass.recipient.name'])]['pass.recipient.name'].value_counts()
    other_fk_targets = other_free_kicks[pd.notna(other_free_kicks['pass.recipient.name'])]['pass.recipient.name'].value_counts()
    
    print("🎯 CORNER TARGETS (Top 5):")
    for i, (player, count) in enumerate(corner_targets.head().items(), 1):
        print(f"  {i}. {player}: {count} receptions")
    
    print("\n🎯 ATTACKING FREE-KICK TARGETS (Top 5):")
    for i, (player, count) in enumerate(attacking_fk_targets.head().items(), 1):
        print(f"  {i}. {player}: {count} receptions")
    
    print("\n🎯 OTHER FREE-KICK TARGETS (Top 5):")
    for i, (player, count) in enumerate(other_fk_targets.head().items(), 1):
        print(f"  {i}. {player}: {count} receptions")
    
    # Create comprehensive visualization using Plotly
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Top Corner Targets', 'Top Attacking Free-Kick Targets', 'Top Other Free-Kick Targets'),
        specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
    )
    
    # Corner targets
    if len(corner_targets) > 0:
        fig.add_trace(
            go.Bar(
                x=corner_targets.head(8).index,
                y=corner_targets.head(8).values,
                name='Corner Targets',
                marker_color='gold',
                marker_line_color='orange',
                marker_line_width=1,
                text=corner_targets.head(8).values,
                textposition='auto',
                textfont=dict(size=10)
            ),
            row=1, col=1
        )
    
    # Attacking free-kick targets
    if len(attacking_fk_targets) > 0:
        fig.add_trace(
            go.Bar(
                x=attacking_fk_targets.head(8).index,
                y=attacking_fk_targets.head(8).values,
                name='Attacking Free-Kick Targets',
                marker_color='red',
                marker_line_color='darkred',
                marker_line_width=1,
                text=attacking_fk_targets.head(8).values,
                textposition='auto',
                textfont=dict(size=10)
            ),
            row=1, col=2
        )
    
    # Other free-kick targets
    if len(other_fk_targets) > 0:
        fig.add_trace(
            go.Bar(
                x=other_fk_targets.head(8).index,
                y=other_fk_targets.head(8).values,
                name='Other Free-Kick Targets',
                marker_color='blue',
                marker_line_color='darkblue',
                marker_line_width=1,
                text=other_fk_targets.head(8).values,
                textposition='auto',
                textfont=dict(size=10)
            ),
            row=1, col=3
        )
    
    # Update layout
    fig.update_layout(
        title_text="Set-Piece Targets by Type",
        title_x=0.5,
        height=500,
        showlegend=False
    )
    
    # Update x-axis labels for better readability
    fig.update_xaxes(tickangle=45)
    
    # Save and show
    fig.write_html('set_piece_targets_by_type.html')
    fig.show()
    
    return corner_targets, attacking_fk_targets, other_fk_targets

# =============================================================================
# STEP 3: DEFENSIVE SET-PIECE ANALYSIS (DINAMO DEFENDING)
# =============================================================================

def analyze_defensive_set_pieces(set_piece_events):
    """
    Analyze Dinamo's defensive set-piece vulnerabilities
    """
    print("\n=== STEP 3: DEFENSIVE SET-PIECE ANALYSIS ===")
    
    # Filter for opponent set-pieces (Dinamo defending)
    dinamo_defending = set_piece_events[
        (set_piece_events['is_dinamo'] == False) & 
        (set_piece_events['type.primary'].isin(['corner', 'free_kick']))
    ].copy()
    
    print(f"✓ Dinamo defensive set-pieces analyzed: {len(dinamo_defending)}")
    
    return dinamo_defending

def analyze_defensive_vulnerabilities(dinamo_defending, df):
    """
    Analyze defensive system and vulnerabilities with proper heatmap
    """
    print("\n--- Analyzing Defensive Vulnerabilities ---")
    
    # Get all shots conceded from set-pieces
    set_piece_shots_conceded = []
    
    for _, set_piece in dinamo_defending.iterrows():
        possession_id = set_piece['possession.id']
        if pd.notna(possession_id):
            # Find shots in this possession
            possession_shots = df[
                (df['possession.id'] == possession_id) & 
                (df['type.primary'] == 'shot') &
                (df['team.name'] != 'Dinamo Bucureşti')
            ]
            
            for _, shot in possession_shots.iterrows():
                set_piece_shots_conceded.append({
                    'x': shot['location.x'],
                    'y': shot['location.y'],
                    'is_goal': shot['shot.isGoal'],
                    'xg': shot['shot.xg'] if pd.notna(shot['shot.xg']) else 0,
                    'set_piece_type': set_piece['type.primary']
                })
    
    if set_piece_shots_conceded:
        shots_df = pd.DataFrame(set_piece_shots_conceded)
        
        # Create defensive vulnerability heatmap using Plotly with custom pitch
        pitch_elements = create_custom_pitch(title="Defensive Set-Piece Vulnerabilities")
        
        # Create heatmap using Plotly
        fig = go.Figure()
        
        # Add pitch elements
        for element in pitch_elements:
            fig.add_trace(element)
        
        # Create heatmap of shots conceded
        if len(shots_df) > 1:  # Need at least 2 points for heatmap
            # Create 2D histogram for heatmap effect
            x_bins = np.linspace(0, 100, 25)
            y_bins = np.linspace(0, 100, 25)
            
            H, xedges, yedges = np.histogram2d(
                shots_df['x'], shots_df['y'], 
                bins=[x_bins, y_bins]
            )
            
            # Create heatmap
            heatmap = go.Heatmap(
                z=H.T,
                x=xedges[:-1],
                y=yedges[:-1],
                colorscale='Reds',
                opacity=0.6,
                showscale=True,
                name='Shot Density'
            )
            fig.add_trace(heatmap)
        
        # Plot individual shots
        goals = shots_df[shots_df['is_goal'] == True]
        non_goals = shots_df[shots_df['is_goal'] == False]
        
        if len(goals) > 0:
            goal_points = go.Scatter(
                x=goals['x'],
                y=goals['y'],
                mode='markers',
                marker=dict(
                    symbol='star',
                    size=15,
                    color='red',
                    opacity=0.9
                ),
                name=f'Goals Conceded ({len(goals)})',
                text=[f'Goal {i+1}' for i in range(len(goals))],
                hoverinfo='text'
            )
            fig.add_trace(goal_points)
        
        if len(non_goals) > 0:
            shot_points = go.Scatter(
                x=non_goals['x'],
                y=non_goals['y'],
                mode='markers',
                marker=dict(
                    size=8,
                    color='orange',
                    opacity=0.7
                ),
                name=f'Shots Conceded ({len(non_goals)})',
                text=[f'Shot {i+1}' for i in range(len(non_goals))],
                hoverinfo='text'
            )
            fig.add_trace(shot_points)
        
        # Add vulnerability zones
        vulnerability_text = go.Scatter(
            x=[10], y=[50],
            mode='text',
            text=['HIGH<br>VULNERABILITY'],
            textfont=dict(size=12, color='red'),
            showlegend=False,
            hoverinfo='skip'
        )
        fig.add_trace(vulnerability_text)
        
        # Update layout
        fig.update_layout(
            title="Defensive Set-Piece Vulnerabilities (Shots Conceded - Heatmap)",
            title_x=0.5,
            xaxis=dict(range=[-5, 105], showgrid=False, zeroline=False),
            yaxis=dict(range=[-5, 105], showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1),
            height=700,
            showlegend=True
        )
        
        # Save and show
        fig.write_html('defensive_vulnerabilities_heatmap.html')
        fig.show()
        
        print(f"✓ Defensive vulnerabilities analyzed:")
        print(f"  • Total shots conceded from set-pieces: {len(shots_df)}")
        print(f"  • Goals conceded from set-pieces: {shots_df['is_goal'].sum()}")
        print(f"  • Total xG conceded: {shots_df['xg'].sum():.3f}")
        
        # Zone analysis
        near_post_shots = len(shots_df[shots_df['x'] < 20])
        penalty_area_shots = len(shots_df[(shots_df['x'] >= 20) & (shots_df['x'] <= 80)])
        far_post_shots = len(shots_df[shots_df['x'] > 80])
        
        print(f"  • Near post shots: {near_post_shots}")
        print(f"  • Penalty area shots: {penalty_area_shots}")
        print(f"  • Far post shots: {far_post_shots}")
    
    return set_piece_shots_conceded

def analyze_aerial_duels(dinamo_defending, df):
    """
    Analyze aerial duels and weak points
    """
    print("\n--- Analyzing Aerial Duels ---")
    
    # Find aerial duels lost by Dinamo during opponent set-pieces
    aerial_duels_lost = []
    
    for _, set_piece in dinamo_defending.iterrows():
        possession_id = set_piece['possession.id']
        if pd.notna(possession_id):
            # Find aerial duels in this possession where Dinamo lost
            possession_duels = df[
                (df['possession.id'] == possession_id) & 
                (df['type.primary'] == 'duel') &
                (df['aerialDuel'] == 'loss') &
                (df['team.name'] == 'Dinamo Bucureşti')
            ]
            
            for _, duel in possession_duels.iterrows():
                aerial_duels_lost.append({
                    'player': duel['player.name'],
                    'position': duel['player.position'],
                    'x': duel['location.x'],
                    'y': duel['location.y'],
                    'set_piece_type': set_piece['type.primary']
                })
    
    if aerial_duels_lost:
        duels_df = pd.DataFrame(aerial_duels_lost)
        
        # Player analysis
        player_duels_lost = duels_df['player'].value_counts()
        
        print("🥊 PLAYERS WITH MOST AERIAL DUELS LOST IN SET-PIECES:")
        for i, (player, count) in enumerate(player_duels_lost.head(5).items(), 1):
            print(f"  {i}. {player}: {count} duels lost")
        
        # Position analysis
        position_duels_lost = duels_df['position'].value_counts()
        
        print(f"\n📍 POSITION ANALYSIS:")
        for position, count in position_duels_lost.items():
            print(f"  • {position}: {count} duels lost")
        
        # Create visualization using Plotly
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Players with Most Aerial Duels Lost', 'Aerial Duels Lost by Position'),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Player chart
        if len(player_duels_lost) > 0:
            fig.add_trace(
                go.Bar(
                    x=player_duels_lost.head(8).index,
                    y=player_duels_lost.head(8).values,
                    name='Players',
                    marker_color='lightcoral',
                    marker_line_color='darkred',
                    marker_line_width=1,
                    text=player_duels_lost.head(8).values,
                    textposition='auto',
                    textfont=dict(size=10)
                ),
                row=1, col=1
            )
        
        # Position chart
        if len(position_duels_lost) > 0:
            fig.add_trace(
                go.Bar(
                    x=position_duels_lost.index,
                    y=position_duels_lost.values,
                    name='Positions',
                    marker_color='lightblue',
                    marker_line_color='darkblue',
                    marker_line_width=1,
                    text=position_duels_lost.values,
                    textposition='auto',
                    textfont=dict(size=10)
                ),
                row=1, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="Aerial Duels Analysis",
            title_x=0.5,
            height=500,
            showlegend=False
        )
        
        # Update x-axis labels for better readability
        fig.update_xaxes(tickangle=45)
        
        # Save and show
        fig.write_html('aerial_duels_analysis.html')
        fig.show()
        
        print(f"✓ Aerial duel analysis completed:")
        print(f"  • Total aerial duels lost: {len(duels_df)}")
        print(f"  • Most vulnerable position: {position_duels_lost.index[0]}")
        print(f"  • Most vulnerable player: {player_duels_lost.index[0]}")
    
    return aerial_duels_lost

def analyze_second_ball_reaction(dinamo_defending, df):
    """
    Analyze second ball reaction and possession recovery
    """
    print("\n--- Analyzing Second Ball Reaction ---")
    
    # Analyze how quickly Dinamo regains possession after clearing set-pieces
    second_ball_analysis = []
    
    for _, set_piece in dinamo_defending.iterrows():
        possession_id = set_piece['possession.id']
        if pd.notna(possession_id):
            # Find the next possession after this set-piece
            set_piece_time = set_piece['minute'] * 60 + set_piece['second']
            
            # Look for Dinamo possession events shortly after
            subsequent_events = df[
                (df['team.name'] == 'Dinamo Bucureşti') &
                (df['minute'] * 60 + df['second'] > set_piece_time) &
                (df['minute'] * 60 + df['second'] <= set_piece_time + 30)  # Within 30 seconds
            ]
            
            if len(subsequent_events) > 0:
                time_to_recovery = (subsequent_events.iloc[0]['minute'] * 60 + 
                                  subsequent_events.iloc[0]['second']) - set_piece_time
                
                second_ball_analysis.append({
                    'set_piece_type': set_piece['type.primary'],
                    'time_to_recovery': time_to_recovery,
                    'recovery_event': subsequent_events.iloc[0]['type.primary']
                })
    
    if second_ball_analysis:
        recovery_df = pd.DataFrame(second_ball_analysis)
        
        print(f"✓ Second ball reaction analysis completed:")
        print(f"  • Average time to recovery: {recovery_df['time_to_recovery'].mean():.1f} seconds")
        print(f"  • Recovery events analyzed: {len(recovery_df)}")
        
        # Recovery time distribution
        fast_recovery = len(recovery_df[recovery_df['time_to_recovery'] <= 10])
        medium_recovery = len(recovery_df[(recovery_df['time_to_recovery'] > 10) & 
                                        (recovery_df['time_to_recovery'] <= 20)])
        slow_recovery = len(recovery_df[recovery_df['time_to_recovery'] > 20])
        
        print(f"  • Fast recovery (≤10s): {fast_recovery} ({fast_recovery/len(recovery_df)*100:.1f}%)")
        print(f"  • Medium recovery (11-20s): {medium_recovery} ({medium_recovery/len(recovery_df)*100:.1f}%)")
        print(f"  • Slow recovery (>20s): {slow_recovery} ({slow_recovery/len(recovery_df)*100:.1f}%)")
        
        # Create visualization using Plotly
        fig = go.Figure()
        
        recovery_categories = ['Fast (≤10s)', 'Medium (11-20s)', 'Slow (>20s)']
        recovery_counts = [fast_recovery, medium_recovery, slow_recovery]
        colors = ['green', 'yellow', 'red']
        
        fig.add_trace(go.Bar(
            x=recovery_categories,
            y=recovery_counts,
            marker_color=colors,
            marker_line_color='black',
            marker_line_width=2,
            text=recovery_counts,
            textposition='auto',
            textfont=dict(size=14, color='black')
        ))
        
        fig.update_layout(
            title="Second Ball Recovery Speed Analysis",
            title_x=0.5,
            xaxis_title="Recovery Speed",
            yaxis_title="Number of Recoveries",
            height=500,
            showlegend=False
        )
        
        # Save and show
        fig.write_html('second_ball_recovery.html')
        fig.show()
    
    return second_ball_analysis

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main execution function
    """
    print("🏆 DINAMO BUCUREȘTI SET-PIECE TACTICAL ANALYSIS")
    print("=" * 60)
    print("Analyzing set-piece strategies for the 2024/25 season...")
    print("=" * 60)
    
    # Step 1: Data Preparation
    df = load_and_prepare_data()
    if df is None:
        return
    
    set_piece_events = filter_set_pieces(df)
    set_piece_events = clean_set_piece_data(set_piece_events)
    
    # Step 2: Offensive Analysis
    dinamo_attacking = analyze_offensive_set_pieces(set_piece_events)
    corner_takers, attacking_dangerous_fk, attacking_non_dangerous_fk, other_fk = identify_main_takers(dinamo_attacking)
    analyze_delivery_zones(dinamo_attacking)
    detect_attacking_patterns(dinamo_attacking)
    corner_targets, attacking_fk_targets, other_fk_targets = identify_dangerous_players(dinamo_attacking)
    
    # Step 3: Defensive Analysis
    dinamo_defending = analyze_defensive_set_pieces(set_piece_events)
    set_piece_shots_conceded = analyze_defensive_vulnerabilities(dinamo_defending, df)
    aerial_duels_lost = analyze_aerial_duels(dinamo_defending, df)
    second_ball_analysis = analyze_second_ball_reaction(dinamo_defending, df)
    
    print("\n" + "="*60)
    print("🎯 ANALYSIS COMPLETE - PROFESSIONAL REPORT READY")
    print("="*60)
    print("✓ All visualizations have been saved as high-quality PNG files")
    print("✓ Comprehensive set-piece analysis completed")
    print("✓ Professional football pitch visualizations generated")
    print("✓ Data-driven insights ready for coaching staff review")

if __name__ == "__main__":
    main()
