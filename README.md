# Dinamo București Set-Piece Tactical Analysis

## 🏆 Project Overview

This project provides a comprehensive tactical analysis of Dinamo București's set-piece strategies for the 2024/25 season. The analysis identifies offensive threats and defensive vulnerabilities to support coaching staff in match preparation.

## 📊 Analysis Objectives

### Primary Goals:
- **Offensive Analysis**: Understand Dinamo's set-piece attacking strategies
- **Defensive Analysis**: Identify vulnerabilities when Dinamo defends set-pieces
- **Player Analysis**: Identify key specialists and dangerous players
- **Tactical Patterns**: Detect common routines and delivery preferences

### Research Questions:
1. Who are Dinamo's main set-piece takers?
2. What are their preferred delivery zones and patterns?
3. Which players are most dangerous in set-piece situations?
4. Where are Dinamo's defensive vulnerabilities?
5. How do they react to second balls after clearing set-pieces?

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Required packages (see requirements.txt)

### Installation
```bash
# Install required packages
pip install -r requirements.txt
```

### Usage
```bash
# Run the complete analysis
python explore_data.py
```

## 📁 Project Structure

```
Proiect/
├── explore_data.py              # Main analysis script
├── Dinamo_Bucuresti_2024_2025_events.csv  # Raw event data
├── requirements.txt             # Python dependencies
├── README.md                   # This file
└── Generated Outputs/          # Analysis results and visualizations
    ├── set_piece_takers.png
    ├── corner_delivery_zones.png
    ├── free_kick_delivery_zones.png
    ├── corner_strategy_analysis.png
    ├── top_set_piece_targets.png
    ├── defensive_vulnerabilities.png
    ├── aerial_duels_analysis.png
    └── second_ball_recovery.png
```

## 🔍 Analysis Methodology

### Step 1: Data Preparation
- Load and validate the event dataset
- Filter for set-piece events (corners, free-kicks, throw-ins, goal-kicks)
- Clean and prepare data for analysis
- Add team context (Dinamo attacking vs. defending)

### Step 2: Offensive Set-Piece Analysis
- **Main Takers Identification**: Analyze who takes corners and free-kicks
- **Delivery Zone Analysis**: Map target areas and delivery preferences
- **Pattern Detection**: Identify short corners vs. direct deliveries
- **Dangerous Players**: Find primary targets and aerial threats

### Step 3: Defensive Set-Piece Analysis
- **Vulnerability Mapping**: Analyze shots conceded from set-pieces
- **Aerial Duel Analysis**: Identify players who lose key defensive headers
- **Second Ball Reaction**: Measure recovery speed after clearing set-pieces

### Step 4: Tactical Summary
- **Top 3 Offensive Threats**: Key players and strategies to watch
- **Top 3 Defensive Vulnerabilities**: Areas to exploit
- **Overall Recommendation**: High-level tactical advice

## 📈 Key Outputs

### Visualizations Generated:
1. **Set-Piece Takers**: Bar charts of corner and free-kick specialists
2. **Delivery Zones**: Pitch maps showing target areas
3. **Strategy Analysis**: Short vs. direct corner delivery breakdown
4. **Target Players**: Top set-piece recipients
5. **Defensive Vulnerabilities**: Heatmap of shots conceded
6. **Aerial Duels**: Player and position vulnerability analysis
7. **Second Ball Recovery**: Speed distribution analysis

### Text Reports:
- Comprehensive tactical insights
- Player performance metrics
- Defensive vulnerability assessment
- Actionable coaching recommendations

## 🎯 Expected Insights

Based on the analysis, you can expect to discover:

### Offensive Strengths:
- Primary set-piece specialists
- Preferred delivery zones
- Most dangerous aerial targets
- Effective attacking patterns

### Defensive Weaknesses:
- Vulnerable areas in the penalty box
- Players who struggle in aerial duels
- Second ball reaction weaknesses
- Set-piece defensive organization issues

## 🔧 Customization

### Modifying Analysis Parameters:
- **Set-piece types**: Edit `set_piece_types` list in the script
- **Zone definitions**: Adjust coordinate boundaries for pitch zones
- **Time thresholds**: Modify second ball recovery time limits
- **Visualization styles**: Customize colors, sizes, and layouts

### Adding New Analysis:
- Extend functions to include additional metrics
- Add new visualization types
- Incorporate additional data sources
- Create comparative analysis with other teams

## 📊 Data Requirements

The script expects a CSV file with the following key columns:
- `type.primary`: Event type (corner, free_kick, etc.)
- `team.name`: Team name for filtering
- `player.name`: Player identification
- `location.x`, `location.y`: Pitch coordinates
- `possession.id`: Possession tracking
- `possession.attack.withShot`: Shot outcomes
- `possession.attack.withGoal`: Goal outcomes
- `possession.attack.xg`: Expected goals

## 🚨 Troubleshooting

### Common Issues:
1. **File not found**: Ensure CSV file is in the same directory
2. **Missing dependencies**: Install required packages with pip
3. **Memory issues**: The script handles large datasets efficiently
4. **Visualization errors**: Check matplotlib backend compatibility

### Performance Tips:
- The script is optimized for large event datasets
- Visualizations are automatically saved as high-quality PNG files
- Progress indicators show analysis completion status

## 📞 Support

For technical support or analysis questions:
- Check the generated error messages
- Verify data file format and location
- Ensure all dependencies are installed
- Review the analysis methodology section

## 📝 License

This project is designed for tactical analysis and coaching purposes. Please ensure compliance with data usage agreements and respect for team privacy.

---

**Ready to analyze Dinamo București's set-piece strategies? Run `python explore_data.py` and discover the tactical insights!** 🏆
