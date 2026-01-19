#!/usr/bin/env python3
"""Plot simple variant experiment figures."""

import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
import fire


def main_exp(data_dir="data/simple_variant"):
    """
    Plot newdemo_figure1.csv temperature data.

    Saves PDF to figures/newdemo_figuremain.pdf
    """
    data_path = Path(data_dir) / "newdemo_figure1.csv"

    if not data_path.exists():
        print(f"Error: Data file '{data_path}' not found.")
        return

    # Read CSV - skip first 3 header rows, parse dates
    df = pd.read_csv(data_path, skiprows=3, header=None,
                     names=['datetime_str', 'datetime_num', 'A', 'B', 'air'],
                     encoding='latin-1')

    # Parse datetime
    df['datetime'] = pd.to_datetime(df['datetime_str'], format='%Y-%m-%d %H:%M:%S')

    # Convert temperature columns to numeric, coercing errors to NaN
    for col in ['A', 'B', 'air']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Filter to x-axis range
    start_time = pd.to_datetime('2025-10-03 14:20:00')
    end_time = pd.to_datetime('2025-10-03 14:51:53')
    df = df[(df['datetime'] >= start_time) & (df['datetime'] <= end_time)]

    # Convert datetime to decimal hours for clean PDF x-axis ticks
    df['time_hours'] = df['datetime'].dt.hour + df['datetime'].dt.minute / 60 + df['datetime'].dt.second / 3600

    # Define segment boundaries (in decimal hours)
    t1 = 14 + 33/60 + 2/3600   # 14:33:02
    t2 = 14 + 42/60 + 42/3600  # 14:42:42

    # Colors
    COLOR_BORO = '#ff7e00'   # Orange
    COLOR_CAF2 = '#00b7ef'   # Blue
    COLOR_OPEN = '#666666'   # Gray
    COLOR_AIR = '#ff69b4'    # Pink

    # Split data into 3 segments
    seg1 = df[df['time_hours'] < t1].copy()
    seg2 = df[(df['time_hours'] >= t1) & (df['time_hours'] < t2)].copy()
    seg3 = df[df['time_hours'] >= t2].copy()

    # Create figure
    fig = go.Figure()

    # Legend 1 (left): Boro, CaF2
    # Legend 2 (right): Uncovered, Ambient

    # Segment 1: Both A and B are "Uncovered" (gray)
    trace = go.Scatter(
        x=seg1['time_hours'], y=seg1['A'],
        mode='lines', name='Uncovered', legendgroup='Uncovered',
        line=dict(color=COLOR_OPEN, width=6),
        legendrank=1,  # Appear first in legend
    )
    trace.legend = 'legend2'
    fig.add_trace(trace)
    trace = go.Scatter(
        x=seg1['time_hours'], y=seg1['B'],
        mode='lines', name='Uncovered', legendgroup='Uncovered', showlegend=False,
        line=dict(color=COLOR_OPEN, width=6),
    )
    trace.legend = 'legend2'
    fig.add_trace(trace)

    # Segment 2: A=CaF2 (blue), B=Boro (orange) - add Boro first for legend order
    trace = go.Scatter(
        x=seg2['time_hours'], y=seg2['B'],
        mode='lines', name='Boro', legendgroup='Boro',
        line=dict(color=COLOR_BORO, width=6),
    )
    trace.legend = 'legend'
    fig.add_trace(trace)
    trace = go.Scatter(
        x=seg2['time_hours'], y=seg2['A'],
        mode='lines', name='CaF2', legendgroup='CaF2',
        line=dict(color=COLOR_CAF2, width=6),
    )
    trace.legend = 'legend'
    fig.add_trace(trace)

    # Segment 3: A=Boro (orange), B=CaF2 (blue)
    trace = go.Scatter(
        x=seg3['time_hours'], y=seg3['A'],
        mode='lines', name='Boro', legendgroup='Boro', showlegend=False,
        line=dict(color=COLOR_BORO, width=6),
    )
    trace.legend = 'legend'
    fig.add_trace(trace)
    trace = go.Scatter(
        x=seg3['time_hours'], y=seg3['B'],
        mode='lines', name='CaF2', legendgroup='CaF2', showlegend=False,
        line=dict(color=COLOR_CAF2, width=6),
    )
    trace.legend = 'legend'
    fig.add_trace(trace)

    # Scale ambient data to primary y-axis coordinates so it renders in same layer
    # Primary y: 51-56, Secondary y: 33-38
    y1_min, y1_max = 51, 56
    y2_min, y2_max = 33.0, 38.0
    ambient_scaled = y1_min + (df['air'] - y2_min) * (y1_max - y1_min) / (y2_max - y2_min)

    # Add Ambient trace FIRST on primary y-axis (scaled) so it renders behind
    trace = go.Scatter(
        x=df['time_hours'],
        y=ambient_scaled,
        mode='lines',
        name='Ambient',
        line=dict(color=COLOR_AIR, width=6),
        opacity=0.7,
        legendrank=2,  # Appear second in legend
    )
    trace.legend = 'legend2'
    fig.add_trace(trace)

    # Reorder: put Ambient first so it renders behind all others
    fig.data = (fig.data[-1],) + fig.data[:-1]

    # Add invisible trace on yaxis2 to force secondary y-axis to display
    fig.add_trace(go.Scatter(
        x=[df['time_hours'].iloc[0]],
        y=[df['air'].iloc[0]],
        mode='markers',
        marker=dict(opacity=0),
        yaxis='y2',
        showlegend=False,
        hoverinfo='skip',
    ))

    # Update layout
    fig.update_layout(
        font=dict(family="Times New Roman, Times, serif", color="black"),
        plot_bgcolor='#f5f5f5',
        paper_bgcolor='white',
        width=1200,
        height=540,
        hovermode='x unified',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=0.0,
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=35),
            itemwidth=50,
            traceorder="normal",
        ),
        legend2=dict(
            orientation="v",
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=0.147,
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=35),
            itemwidth=50,
            traceorder="normal",
        ),
        margin=dict(l=100, r=80, t=40, b=130),
        xaxis=dict(
            title=dict(text="Time", font=dict(size=42)),
            tickfont=dict(size=39),
            showgrid=True,
            gridcolor='rgba(200,200,200,0.5)',
            gridwidth=1,
            tickmode='array',
            tickvals=[14 + 20/60, 14 + 30/60, 14 + 40/60, 14 + 50/60],
            ticktext=['14:20', '14:30', '14:40', '14:50'],
            range=[14 + 20/60, 14 + 52/60],
        ),
        yaxis=dict(
            title=dict(text="Surf Temp (°C)", font=dict(size=42)),
            tickfont=dict(size=39),
            showgrid=True,
            gridcolor='rgba(200,200,200,0.5)',
            gridwidth=1,
            range=[51, 56],
        ),
        yaxis2=dict(
            title=dict(text="Air Temp (°C)", font=dict(size=42)),
            tickfont=dict(size=39),
            overlaying='y',
            side='right',
            range=[33.0, 38.0],
            showgrid=False,
            dtick=1,
        ),
    )

    # Add vertical lines at key times
    fig.add_vline(x=14 + 33/60 + 2/3600, line_width=6, line_color="black")
    fig.add_vline(x=14 + 42/60 + 42/3600, line_width=6, line_color="black")

    # Add segment labels (a), (b), (c) centered in each segment, just above bottom
    x_start = 14 + 20/60
    x_end = 14 + 52/60
    seg1_center = (x_start + t1) / 2
    seg2_center = (t1 + t2) / 2
    seg3_center = (t2 + x_end) / 2
    label_y = 51.0  # Right at the y-axis minimum

    for label, x_pos in [('(a)', seg1_center), ('(b)', seg2_center), ('(c)', seg3_center)]:
        fig.add_annotation(
            x=x_pos, y=label_y,
            text=label,
            showarrow=False,
            font=dict(size=36, family="Times New Roman, Times, serif", color="black"),
            xanchor='center',
            yanchor='bottom',
        )

    # Save to figures directory
    figures_dir = Path(__file__).parent / 'figures'
    figures_dir.mkdir(exist_ok=True)

    pdf_file = figures_dir / 'newdemo_figuremain.pdf'
    fig.write_image(str(pdf_file), format='pdf', width=1200, height=540)
    print(f"Saved: {pdf_file}")


def glass_exp(data_dir="data/simple_variant"):
    """
    Plot newdemo_figure1.csv temperature data for glass experiment.

    Saves PDF to figures/newdemo_figureglass.pdf
    """
    data_path = Path(data_dir) / "newdemo_figure1.csv"

    if not data_path.exists():
        print(f"Error: Data file '{data_path}' not found.")
        return

    # Read CSV - skip first 3 header rows, parse dates
    df = pd.read_csv(data_path, skiprows=3, header=None,
                     names=['datetime_str', 'datetime_num', 'A', 'B', 'air'],
                     encoding='latin-1')

    # Parse datetime
    df['datetime'] = pd.to_datetime(df['datetime_str'], format='%Y-%m-%d %H:%M:%S')

    # Convert temperature columns to numeric, coercing errors to NaN
    for col in ['A', 'B', 'air']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Filter to x-axis range
    start_time = pd.to_datetime('2025-10-03 15:22:26')
    end_time = pd.to_datetime('2025-10-03 15:39:35')
    df = df[(df['datetime'] >= start_time) & (df['datetime'] <= end_time)]

    # Convert datetime to decimal hours for clean PDF x-axis ticks
    df['time_hours'] = df['datetime'].dt.hour + df['datetime'].dt.minute / 60 + df['datetime'].dt.second / 3600

    # Define segment boundary (in decimal hours)
    t1 = 15 + 32/60 + 2/3600   # 15:32:02

    # Colors
    COLOR_BORO = '#ff7e00'   # Orange
    COLOR_CAF2 = '#00b7ef'   # Blue
    COLOR_AIR = '#ff69b4'    # Pink

    # Split data into 2 segments
    seg1 = df[df['time_hours'] < t1].copy()
    seg2 = df[df['time_hours'] >= t1].copy()

    # Create figure
    fig = go.Figure()

    # Segment 1: A=Boro (orange), B=CaF2 (blue)
    trace = go.Scatter(
        x=seg1['time_hours'], y=seg1['A'],
        mode='lines', name='Boro', legendgroup='Boro',
        line=dict(color=COLOR_BORO, width=6),
    )
    trace.legend = 'legend'
    fig.add_trace(trace)
    trace = go.Scatter(
        x=seg1['time_hours'], y=seg1['B'],
        mode='lines', name='CaF2', legendgroup='CaF2',
        line=dict(color=COLOR_CAF2, width=6),
    )
    trace.legend = 'legend'
    fig.add_trace(trace)

    # Segment 2: A=CaF2 (blue), B=Boro (orange) - vice versa
    trace = go.Scatter(
        x=seg2['time_hours'], y=seg2['A'],
        mode='lines', name='CaF2', legendgroup='CaF2', showlegend=False,
        line=dict(color=COLOR_CAF2, width=6),
    )
    trace.legend = 'legend'
    fig.add_trace(trace)
    trace = go.Scatter(
        x=seg2['time_hours'], y=seg2['B'],
        mode='lines', name='Boro', legendgroup='Boro', showlegend=False,
        line=dict(color=COLOR_BORO, width=6),
    )
    trace.legend = 'legend'
    fig.add_trace(trace)

    # Scale ambient data to primary y-axis coordinates so it renders in same layer
    # Primary y: 59-63.5, Secondary y: 35.5-40
    y1_min, y1_max = 59, 63.5
    y2_min, y2_max = 35.5, 40.0
    ambient_scaled = y1_min + (df['air'] - y2_min) * (y1_max - y1_min) / (y2_max - y2_min)

    # Add Ambient trace on primary y-axis (scaled) so it renders behind
    trace = go.Scatter(
        x=df['time_hours'],
        y=ambient_scaled,
        mode='lines',
        name='Ambient',
        line=dict(color=COLOR_AIR, width=6),
        opacity=0.7,
        legendrank=2,
    )
    trace.legend = 'legend2'
    fig.add_trace(trace)

    # Reorder: put Ambient first so it renders behind all others
    fig.data = (fig.data[-1],) + fig.data[:-1]

    # Add invisible trace on yaxis2 to force secondary y-axis to display
    fig.add_trace(go.Scatter(
        x=[df['time_hours'].iloc[0]],
        y=[df['air'].iloc[0]],
        mode='markers',
        marker=dict(opacity=0),
        yaxis='y2',
        showlegend=False,
        hoverinfo='skip',
    ))

    # X-axis range
    x_start = 15 + 22/60 + 26/3600
    x_end = 15 + 39/60 + 35/3600

    # Update layout
    fig.update_layout(
        font=dict(family="Times New Roman, Times, serif", color="black"),
        plot_bgcolor='#f5f5f5',
        paper_bgcolor='white',
        width=1200,
        height=540,
        hovermode='x unified',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=0.0,
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=35),
            itemwidth=50,
            traceorder="normal",
        ),
        legend2=dict(
            orientation="v",
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=0.147,
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=35),
            itemwidth=50,
            traceorder="normal",
        ),
        margin=dict(l=100, r=80, t=40, b=130),
        xaxis=dict(
            title=dict(text="Time", font=dict(size=42)),
            tickfont=dict(size=39),
            showgrid=True,
            gridcolor='rgba(200,200,200,0.5)',
            gridwidth=1,
            tickmode='array',
            tickvals=[15 + 25/60, 15 + 30/60, 15 + 35/60],
            ticktext=['15:25', '15:30', '15:35'],
            range=[x_start, x_end],
        ),
        yaxis=dict(
            title=dict(text="Surf Temp (°C)", font=dict(size=42)),
            tickfont=dict(size=39),
            showgrid=True,
            gridcolor='rgba(200,200,200,0.5)',
            gridwidth=1,
            range=[59, 63.5],
            dtick=1,
        ),
        yaxis2=dict(
            title=dict(text="Air Temp (°C)", font=dict(size=42)),
            tickfont=dict(size=39),
            overlaying='y',
            side='right',
            range=[35.5, 40.0],
            showgrid=False,
            tickmode='array',
            tickvals=[35.5, 36.5, 37.5, 38.5, 39.5],
        ),
    )

    # Add vertical line at segment boundary
    fig.add_vline(x=t1, line_width=6, line_color="black")

    # Add segment labels (a), (b) centered in each segment, at bottom
    seg1_center = (x_start + t1) / 2
    seg2_center = (t1 + x_end) / 2
    label_y = 59.0  # Right at the y-axis minimum (59)

    for label, x_pos in [('(a)', seg1_center), ('(b)', seg2_center)]:
        fig.add_annotation(
            x=x_pos, y=label_y,
            text=label,
            showarrow=False,
            font=dict(size=36, family="Times New Roman, Times, serif", color="black"),
            xanchor='center',
            yanchor='bottom',
        )

    # Save to figures directory
    figures_dir = Path(__file__).parent / 'figures'
    figures_dir.mkdir(exist_ok=True)

    pdf_file = figures_dir / 'newdemo_figureglass.pdf'
    fig.write_image(str(pdf_file), format='pdf', width=1200, height=540)
    print(f"Saved: {pdf_file}")


if __name__ == '__main__':
    fire.Fire({
        'main_exp': main_exp,
        'glass_exp': glass_exp,
    })
