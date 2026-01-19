import os

import pandas as pd
import numpy as np
import sys
import subprocess
import webbrowser
import tempfile
from pathlib import Path
import plotly.graph_objects as go
import tabulate
from plotly.subplots import make_subplots
import fire
import time
from ghe_exp import graph_config

CONDITION_RANGES = None
DROP_RANGES = None

# Check for dual y-axis aluminum-acrylic environment variable
DUAL_YAXIS_ALAC = os.environ.get('DUAL_YAXIS_ALAC') == '1'


def load_ranges():
    """Load drop and condition ranges from CSV files"""
    import csv
    global CONDITION_RANGES, DROP_RANGES

    DROP_RANGES = []
    CONDITION_RANGES = []

    with open('data/main_exp/drop_ranges.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['start'].startswith("#"):
                # comment, ignore
                continue

            columns = 'all' if row['columns'] == 'all' else eval(row['columns'])
            DROP_RANGES.append({
                'start': row['start'],
                'end': row['end'],
                'columns': columns,
                'reason': row['reason']
            })

    with open('data/main_exp/condition_ranges.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            CONDITION_RANGES.append({
                'start': row['start'],
                'end': row['end'],
                'condition': row['condition']
            })

    # Add adjustment periods from drop ranges to conditions
    for ent in DROP_RANGES:
        if 'adjustment' in ent['reason']:
            CONDITION_RANGES.append({
                'start': ent['start'],
                'end': ent['end'],
                'condition': "adjustment period",
            })


load_ranges()


def load_sun_data():
    """Load and parse sun data from sun_2025.csv"""
    script_dir = Path(__file__).parent
    sun_file = script_dir / 'data/main_exp/sun_2025.csv'
    
    if not sun_file.exists():
        print(f"Warning: Sun data file not found at {sun_file}")
        return {}
    
    try:
        sun_df = pd.read_csv(sun_file)
        sun_data = {}
        
        for _, row in sun_df.iterrows():
            date_str = str(row['date_yyyymmdd'])
            solar_noon_str = str(row['solar_noon_hhmm']).zfill(4)
            
            date = pd.to_datetime(date_str).date()
            hour = int(solar_noon_str[:2])
            minute = int(solar_noon_str[2:])
            
            sun_data[date] = {'solar_noon_hour': hour, 'solar_noon_minute': minute}
        
        return sun_data
    except Exception as e:
        print(f"Warning: Failed to load sun data: {e}")
        return {}


def get_solar_noon_datetime(date, sun_data):
    """Get solar noon datetime for a specific date"""
    if date not in sun_data:
        return None
    
    solar_data = sun_data[date]
    return pd.Timestamp(
        year=date.year,
        month=date.month, 
        day=date.day,
        hour=solar_data['solar_noon_hour'], 
        minute=solar_data['solar_noon_minute']
    )


def detect_encoding(csv_file):
    """Detect file encoding using uchardet command"""
    try:
        result = subprocess.run(['uchardet', str(csv_file)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            encoding = result.stdout.decode('utf8').strip()
            print("Detected encoding:", encoding)
            return encoding
    except:
        pass
    print("Warning: encoding detection failed, falling back to latin-1")
    return 'latin-1'


def read_roofalum_csv(csv_file):
    """Read CSV file with 3-header-row format and return processed dataframe"""
    encoding = detect_encoding(csv_file)
    df = pd.read_csv(csv_file, skiprows=3, encoding=encoding)

    # Check if we have temperature-only data (6 columns) vs full data (10 columns)
    if len(df.columns) == 6:
        # Temperature data only - rename columns and add missing IR columns with NaN
        df.columns = ['datetime1', 'datetime2', 'air', 'A', 'B', 'C']
        # Add missing IR columns filled with NaN
        df['solar'] = np.nan
        df['ir_net'] = np.nan
        df['thermistorV'] = np.nan
        df['excitationV'] = np.nan
    else:
        # Full data - set all column names
        df.columns = ['datetime1', 'datetime2', 'air', 'A', 'B', 'C', 'solar', 'ir_net', 'thermistorV', 'excitationV']

    # Convert numeric columns
    numeric_cols = ['air', 'A', 'B', 'C', 'solar', 'ir_net', 'thermistorV', 'excitationV']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Convert datetime and reorganize columns
    df['datetime'] = pd.to_datetime(df['datetime1'])
    df.drop(columns=['datetime1', 'datetime2'], inplace=True)
    df = df[['datetime'] + [col for col in df.columns if col != 'datetime']]

    # Sort by datetime
    return df.sort_values('datetime').reset_index(drop=True)


def pre_filter_data(df):
    """Apply data filtering based on DROP_RANGES configuration"""
    df_filtered = df.copy()

    for drop_range in DROP_RANGES:
        start_time = pd.to_datetime(drop_range['start'])
        end_time = pd.to_datetime(drop_range['end'])
        columns = drop_range['columns']
        reason = drop_range['reason']

        time_mask = (df_filtered['datetime'] >= start_time) & (df_filtered['datetime'] <= end_time)
        rows_affected = time_mask.sum()

        if rows_affected > 0:
            print(f"Filtering {rows_affected} rows from {start_time} to {end_time}: {reason}")

            if columns == 'all':
                data_columns = ['air', 'A', 'B', 'C', 'solar', 'ir_net', 'thermistorV', 'excitationV']
                for col in data_columns:
                    if col in df_filtered.columns:
                        df_filtered.loc[time_mask, col] = np.nan
            else:
                for col in columns:
                    if col in df_filtered.columns:
                        df_filtered.loc[time_mask, col] = np.nan
                        print(f"  - Dropped {col} data")

    return df_filtered


def calculate_derived_values(df):
    """Calculate derived IR values using thermistor formulas"""
    mask = df['thermistorV'].notna() & df['excitationV'].notna() & (df['excitationV'] != df['thermistorV'])

    # Initialize columns
    derived_cols = ['thermistorR', 'pyrgeTK', 'pyrge_emission', 'air_temp_emission', 'ir_in', 'ir_sky', 'pyrge']
    for col in derived_cols:
        df[col] = np.nan

    PYRGE_K2_FACTOR = 0.97  # 1.028

    if mask.any():
        # Calculate thermistor resistance and temperature
        df.loc[mask, 'thermistorR'] = 24900 * (df.loc[mask, 'thermistorV'] / (df.loc[mask, 'excitationV'] - df.loc[mask, 'thermistorV']))
        ln_R = np.log(df.loc[mask, 'thermistorR'])
        df.loc[mask, 'pyrgeTK'] = 1 / (0.000932794 + 0.000221451 * ln_R + 0.000000126233 * (ln_R**3))
        df.loc[mask, 'pyrge_emission'] = PYRGE_K2_FACTOR * 0.000000056704 * (df.loc[mask, 'pyrgeTK']**4)

        # Calculate ir_in where ir_net is available
        ir_mask = mask & df['ir_net'].notna()
        df.loc[ir_mask, 'ir_in'] = df.loc[ir_mask, 'ir_net'] + df.loc[ir_mask, 'pyrge_emission']
        
        # Convert to Celsius
        df.loc[mask, 'pyrge'] = df.loc[mask, 'pyrgeTK'] - 273.15

    # Calculate air_temp_emission with linear traversal
    sigma = 5.67e-8  # Stefan-Boltzmann constant
    emissivity = 0.95
    last_air_temp = None
    last_air_time = None

    for idx in df.index:
        if pd.notna(df.loc[idx, 'air']):
            last_air_temp = df.loc[idx, 'air']
            last_air_time = df.loc[idx, 'datetime']

        if pd.notna(df.loc[idx, 'ir_in']) and last_air_temp is not None and last_air_time is not None:
            time_diff = abs(df.loc[idx, 'datetime'] - last_air_time)
            if time_diff <= pd.Timedelta(minutes=graph_config.MAX_TIME_DIFF_AIR_TEMP):
                air_temp_k = last_air_temp + 273.15
                df.loc[idx, 'air_temp_emission'] = emissivity * sigma * (air_temp_k ** 4)

    # Calculate ir_sky
    air_sky_mask = df['ir_in'].notna() & df['air_temp_emission'].notna()
    if air_sky_mask.any():
        F_SKY = 0.9148372708415735  # View factor sensor to sky, a 3.81mm radius aperture to a 20mm radius opening 6mm above it  https://sterad.net/
        df.loc[air_sky_mask, 'ir_sky'] = (df.loc[air_sky_mask, 'ir_in'] - df.loc[air_sky_mask, 'air_temp_emission'] * (1 - F_SKY)) / F_SKY

    return df


def wrap_datetime_to_time(dt):
    """Convert datetime to time-only for wrapping"""
    if pd.isna(dt):
        return dt
    today = pd.Timestamp.now().date()
    return pd.Timestamp.combine(today, dt.time())


def add_vrect_to_subplots(fig, x0, x1, color_config):
    """Helper to add vertical rectangle to all subplots"""
    for row in [1, 2, 3, 4, 5]:
        fig.add_vrect(
            x0=x0, x1=x1,
            exclude_empty_subplots=False,
            fillcolor=color_config['color'],
            opacity=color_config['opacity'],
            line_width=0,
            layer="below",
            row=row, col=1
        )


def add_vrect_to_specific_rows(fig, x0, x1, color_config, rows, numeric_x=False, y_range=None):
    """
    Add vertical rectangle to specific subplot rows.

    Args:
        fig: Plotly figure
        x0, x1: Start and end x coordinates (datetime or numeric)
        color_config: Dict with 'color' and 'opacity' keys
        rows: Tuple of row numbers to draw on
        numeric_x: If True, convert datetime to numeric hours (0-24) for PDF export
        y_range: Optional y-axis range; defaults to TEMP_DIFF_Y_RANGE

    When numeric_x=True, uses fig.add_shape with row/col parameters instead of
    fig.add_vrect. This is required for PDF export with kaleido, which doesn't
    properly render vrects on subplots created with make_subplots.

    Midnight wrap fix: When a time range ends at midnight (e.g., 21:05-00:00),
    converting to numeric hours gives x1=0.0 which is less than x0=21.08. This
    would create a shape spanning backwards across the entire plot. We detect
    this (x1 < x0) and set x1=24.0 to represent end-of-day correctly.
    """
    # Convert to numeric hours if requested
    if numeric_x:
        x0 = x0.hour + x0.minute / 60 + x0.second / 3600
        x1 = x1.hour + x1.minute / 60 + x1.second / 3600
        # Handle midnight wrap: if end is before start, it means end wrapped to next day
        if x1 < x0:
            x1 = 24.0
        # Use add_shape for numeric x (works better with PDF export)
        if y_range is None:
            y_range = graph_config.TEMP_DIFF_Y_RANGE
        for row in rows:
            fig.add_shape(
                type="rect",
                x0=x0, x1=x1,
                y0=y_range[0], y1=y_range[1],
                fillcolor=color_config['color'],
                opacity=color_config['opacity'],
                line_width=0,
                layer="below",
                row=row, col=1,
            )
    else:
        for row in rows:
            fig.add_vrect(
                x0=x0, x1=x1,
                exclude_empty_subplots=False,
                fillcolor=color_config['color'],
                opacity=color_config['opacity'],
                line_width=0,
                layer="below",
                row=row, col=1
            )


def aggregate_conditions_per_minute(condition_ranges, df, wrap=False):
    """
    Aggregate conditions per minute and calculate weighted averages.
    
    Returns:
        dict: {minute_timestamp: {'conditions': {condition: count}, 'total': total_count}}
    """
    # Find data bounds
    columns = [col for col in df.columns if 'datetime' not in col]
    has_data_mask = df[columns].notna().any(axis=1)
    
    if not has_data_mask.any():
        return {}
    
    # Initialize minute aggregation
    minute_aggregation = {}
    
    for range_data in condition_ranges:
        if range_data['condition'] == 'adjustment period':
            continue
            
        start_time = pd.to_datetime(range_data['start'])
        end_time = pd.to_datetime(range_data['end'])
        condition = range_data['condition']
        
        # Check if data exists in range
        time_mask = (df['datetime'] >= start_time) & (df['datetime'] <= end_time)
        if not (has_data_mask & time_mask).any():
            continue
        
        if wrap:
            # For wrapped mode, we need to handle ranges that might span midnight
            start_wrapped = wrap_datetime_to_time(start_time)
            end_wrapped = wrap_datetime_to_time(end_time)
            
            if end_wrapped < start_wrapped:
                # Range crosses midnight - split into two parts
                # Part 1: from start to end of day
                current = start_wrapped.floor('T')
                end_of_day = pd.Timestamp.combine(start_wrapped.date(), pd.Timestamp("23:59:00").time())
                
                while current <= end_of_day:
                    minute_key = current
                    if minute_key not in minute_aggregation:
                        minute_aggregation[minute_key] = {'conditions': {}, 'total': 0}
                    
                    if condition not in minute_aggregation[minute_key]['conditions']:
                        minute_aggregation[minute_key]['conditions'][condition] = 0
                    
                    minute_aggregation[minute_key]['conditions'][condition] += 1
                    minute_aggregation[minute_key]['total'] += 1
                    
                    current += pd.Timedelta(minutes=1)
                
                # Part 2: from start of day to end
                current = pd.Timestamp.combine(start_wrapped.date(), pd.Timestamp("00:00:00").time())
                end_minute = end_wrapped.floor('T')
                
                while current <= end_minute:
                    minute_key = current
                    if minute_key not in minute_aggregation:
                        minute_aggregation[minute_key] = {'conditions': {}, 'total': 0}
                    
                    if condition not in minute_aggregation[minute_key]['conditions']:
                        minute_aggregation[minute_key]['conditions'][condition] = 0
                    
                    minute_aggregation[minute_key]['conditions'][condition] += 1
                    minute_aggregation[minute_key]['total'] += 1
                    
                    current += pd.Timedelta(minutes=1)
            else:
                # Range doesn't cross midnight
                current = start_wrapped.floor('T')
                end_minute = end_wrapped.floor('T')
                
                while current <= end_minute:
                    minute_key = current
                    if minute_key not in minute_aggregation:
                        minute_aggregation[minute_key] = {'conditions': {}, 'total': 0}
                    
                    if condition not in minute_aggregation[minute_key]['conditions']:
                        minute_aggregation[minute_key]['conditions'][condition] = 0
                    
                    minute_aggregation[minute_key]['conditions'][condition] += 1
                    minute_aggregation[minute_key]['total'] += 1
                    
                    current += pd.Timedelta(minutes=1)
        else:
            # Non-wrapped mode
            current = start_time.floor('T')  # Floor to minute
            end_minute = end_time.floor('T')
            
            while current <= end_minute:
                minute_key = current
                    
                if minute_key not in minute_aggregation:
                    minute_aggregation[minute_key] = {'conditions': {}, 'total': 0}
                
                if condition not in minute_aggregation[minute_key]['conditions']:
                    minute_aggregation[minute_key]['conditions'][condition] = 0
                
                minute_aggregation[minute_key]['conditions'][condition] += 1
                minute_aggregation[minute_key]['total'] += 1
                
                current += pd.Timedelta(minutes=1)
    
    return minute_aggregation


def add_condition_backgrounds(fig, df, wrap=False, file_condition_ranges=None, row_files_map=None, numeric_x=False, rows=None):
    """
    Add background shading for condition ranges using weighted color averaging.

    Args:
        fig: Plotly figure
        df: DataFrame with datetime column
        wrap: Whether to wrap times to 24-hour format
        file_condition_ranges: Optional dict of condition ranges per file for filtering
        row_files_map: Optional dict mapping row numbers to files for selective filtering
        numeric_x: Whether x-axis uses numeric hours (0-24) instead of timestamps
        rows: Optional tuple of row numbers to draw on (overrides other row logic)
    """
    # Determine which rows to draw on
    if rows is not None:
        # Simple override - draw on specified rows only
        rows_config = {rows: CONDITION_RANGES}
    elif row_files_map and file_condition_ranges:
        # Use selective filtering mode
        rows_config = {}
        
        # Rows 1-2 get all conditions from all files
        all_ranges = []
        for file_ranges in file_condition_ranges.values():
            all_ranges.extend(file_ranges)
        rows_config[(1, 2)] = all_ranges
        
        # Rows 3-5 get conditions only from their specific files
        for row, files in row_files_map.items():
            if row in [3, 4, 5] and files:
                row_ranges = []
                for file in files:
                    if file in file_condition_ranges:
                        row_ranges.extend(file_condition_ranges[file])
                if row_ranges:
                    rows_config[(row,)] = row_ranges
    else:
        # Simple mode - draw on all rows
        rows_config = {(1, 2, 3, 4, 5): CONDITION_RANGES}
    
    # Process each row configuration
    for rows, condition_ranges in rows_config.items():
        if not condition_ranges:
            continue
            
        # Aggregate conditions per minute
        minute_aggregation = aggregate_conditions_per_minute(condition_ranges, df, wrap)

        if not minute_aggregation:
            continue

        # Sort minutes to find contiguous ranges
        sorted_minutes = sorted(minute_aggregation.keys())
        
        # Group contiguous minutes with same condition mix
        i = 0
        while i < len(sorted_minutes):
            start_minute = sorted_minutes[i]
            current_conditions = minute_aggregation[start_minute]['conditions']
            
            # Find end of contiguous range with same conditions
            j = i + 1
            while j < len(sorted_minutes):
                next_minute = sorted_minutes[j]
                next_conditions = minute_aggregation[next_minute]['conditions']
                
                # Check if next minute is contiguous and has same conditions
                if wrap:
                    # For wrapped times, check if minutes are consecutive
                    time_diff = (next_minute - sorted_minutes[j-1]).total_seconds() / 60
                    if time_diff != 1 or current_conditions != next_conditions:
                        break
                else:
                    # For non-wrapped times, check if minutes are consecutive
                    time_diff = (next_minute - sorted_minutes[j-1]).total_seconds() / 60
                    if time_diff != 1 or current_conditions != next_conditions:
                        break
                j += 1
            
            # Draw vrect for this contiguous range
            end_minute = sorted_minutes[j-1]
            
            # Calculate weighted average color
            colors_with_weights = []
            for condition, count in current_conditions.items():
                if condition in graph_config.CONDITION_COLORS:
                    color_config = graph_config.CONDITION_COLORS[condition]
                    colors_with_weights.append((color_config['color'], color_config['opacity'], count))
            
            if colors_with_weights:
                avg_color, avg_opacity = graph_config.average_colors_weighted(colors_with_weights)

                # Add vrect with averaged color
                vrect_config = {'color': avg_color, 'opacity': 1.0}
                
                # Calculate actual end time (add 1 minute to include the full minute)
                end_time = end_minute + pd.Timedelta(minutes=1)
                
                # Draw on specified rows
                add_vrect_to_specific_rows(fig, start_minute, end_time, vrect_config, rows, numeric_x=numeric_x)
            
            i = j


def plot_trace_with_gaps(wrap=False, opacity=1.0, **kwargs):
    """Create multiple go.Scatter traces for contiguous ranges of non-NaN values"""
    x_data = kwargs.get('x')
    y_data = kwargs.get('y')

    if x_data is None or y_data is None:
        return []

    x_array = np.array(x_data)
    y_array = np.array(y_data)
    valid_mask = ~(pd.isna(x_array) | pd.isna(y_array))

    if not valid_mask.any():
        return []

    # Check if x values are numeric (not datetime)
    is_numeric_x = np.issubdtype(x_array.dtype, np.floating) or np.issubdtype(x_array.dtype, np.integer)

    traces = []
    x_array_dt = None if is_numeric_x else pd.to_datetime(x_array, errors='coerce')

    i = 0
    while i < len(valid_mask):
        while i < len(valid_mask) and not valid_mask[i]:
            i += 1

        if i >= len(valid_mask):
            break

        range_start = i
        prev_val = None

        while i < len(valid_mask):
            if valid_mask[i]:
                if is_numeric_x:
                    # For numeric hours: detect wrap when value decreases significantly (day boundary)
                    curr_val = x_array[i]
                    if prev_val is not None and curr_val < prev_val - 1:
                        break
                    prev_val = curr_val
                else:
                    # For datetime: detect wrap from hour 23 to hour 0
                    curr_time = x_array_dt[i]
                    if prev_val is not None and prev_val.hour >= 23 and curr_time.hour <= 0:
                        break
                    prev_val = curr_time
                i += 1
            else:
                nan_start = i
                while i < len(valid_mask) and not valid_mask[i]:
                    i += 1
                nan_count = i - nan_start
                if nan_count >= graph_config.MIN_GAP_SIZE:
                    break

        range_end = nan_start if 'nan_start' in locals() and 'nan_count' in locals() and nan_count >= graph_config.MIN_GAP_SIZE else i

        range_mask = valid_mask[range_start:range_end]
        if range_mask.any():
            segment_kwargs = kwargs.copy()
            segment_kwargs['x'] = x_array[range_start:range_end][range_mask]
            segment_kwargs['y'] = y_array[range_start:range_end][range_mask]

            if len(traces) > 0:
                segment_kwargs['showlegend'] = False

            if 'legendgroup' in kwargs:
                segment_kwargs['legendgroup'] = kwargs['legendgroup']
            elif 'name' in segment_kwargs:
                segment_kwargs['legendgroup'] = segment_kwargs['name']
            
            segment_kwargs['opacity'] = opacity
            traces.append(go.Scatter(**segment_kwargs))

        if 'nan_start' in locals():
            del nan_start
            del nan_count

    return traces


def add_temperature_trace(df, column, name, color, fig, wrap=False, opacity=1.0, row=1, secondary_y=False, width=3):
    """Helper to add temperature traces with common formatting"""
    traces = plot_trace_with_gaps(
        wrap=wrap,
        opacity=opacity,
        x=df['datetime'],
        y=df[column],
        mode='lines',
        name=name,
        line=dict(color=color, width=width),
        hovertemplate=f'{name}: %{{y:.1f}}°C<br>%{{x}}<extra></extra>'
    )
    for trace in traces:
        fig.add_trace(trace, secondary_y=secondary_y, row=row, col=1)


def plot_temperature_difference(df, diff_column, material1, material2, color_key, fig, wrap=False, opacity=1.0, row=2, lw=3, short_names=False, legend_name='legend2'):
    """Helper to plot temperature differences between materials"""
    if diff_column not in df.columns:
        return

    if short_names:
        name = material1
    else:
        name = f'ΔT: {material1} vs. {"Air Temp" if material2 == "air" else material2}'
        if os.environ.get("TWOPANE"):
            if material1 == 'Boro' and material2 == 'CaF2':
                name = 'ΔT: 2xBoro vs. Boro/CaF2'

            if material1 == 'Boro' and material2 == 'Empty':
                name = 'ΔT: 2xBoro vs. 1xBoro'

    # Determine if this should go on secondary y-axis
    use_secondary_y = (DUAL_YAXIS_ALAC and row == 2 and diff_column == 'Aluminum_minus_Acrylic')

    colors = graph_config.TRACE_COLORS
    traces = plot_trace_with_gaps(
        wrap=wrap,
        opacity=opacity,
        x=df['datetime'],
        y=df[diff_column],
        mode='lines',
        name=name,
        line=dict(color=colors[color_key], width=lw),
        hovertemplate=f'{material1} - {material2}: %{{y:.2f}}°C<br>%{{x}}<extra></extra>',
        legendgroup=f'{material1} - {material2}'
    )
    for trace in traces:
        trace.legend = legend_name
        fig.add_trace(trace, row=row, col=1, secondary_y=use_secondary_y)


def add_vertical_lines_to_subplots(fig, time_value, line_name, color='rgba(128, 128, 128, 0.5)', width=2, dash='dash'):
    """Add vertical lines to all subplots"""
    for row in [1, 2, 3, 4, 5]:
        fig.add_vline(
            x=time_value, 
            line_width=width, 
            line_dash=dash, 
            line_color=color,
            row=row, col=1
        )


def get_material_assignments_per_file(csv_files):
    """Get material assignments for each CSV file"""
    file_materials = {}
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        materials = {'A': None, 'B': None, 'C': None}
        for col in df.columns:
            if '/' in col:
                material, channel = col.split('/')
                if channel in ['A', 'B', 'C']:
                    materials[channel] = material
        file_materials[csv_file] = materials
    return file_materials


def get_condition_ranges_per_file(csv_files):
    """Get which condition ranges apply to each CSV file based on data presence"""
    file_condition_ranges = {}
    
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Get data columns (exclude datetime and condition columns)
        data_columns = [col for col in df.columns if col not in ['datetime', 'source_file'] and not col.endswith('_')]
        has_data_mask = df[data_columns].notna().any(axis=1)
        
        applicable_ranges = []
        for range_data in CONDITION_RANGES:
            start_time = pd.to_datetime(range_data['start'])
            end_time = pd.to_datetime(range_data['end'])
            
            # Check if this file has any data in this time range
            time_mask = (df['datetime'] >= start_time) & (df['datetime'] <= end_time)
            if (has_data_mask & time_mask).any():
                applicable_ranges.append(range_data)
        
        file_condition_ranges[csv_file] = applicable_ranges
    
    return file_condition_ranges


def calculate_averaged_temp_diffs(csv_files):
    """Calculate averaged temperature differences across files with wrapped timesteps"""
    print("Processing files for averaging...")
    
    # Get material assignments
    file_materials = get_material_assignments_per_file(csv_files)
    
    # Group data by material arrangements
    arrangement_data = {}  # Key: (matA, matB, matC), Value: list of DataFrames with diff columns
    pair_arrangement_data = {'AB': {}, 'AC': {}, 'BC': {}}  # Key: pair, Value: dict of (mat1, mat2) -> list of DFs
    
    all_diff_columns = set()

    for csv_file in csv_files:
        file_df = pd.read_csv(csv_file)
        file_df['datetime'] = pd.to_datetime(file_df['datetime'])
        
        materials = file_materials[csv_file]
        rev_materials = {v: k for k, v in materials.items() if v is not None}
        print(f"{csv_file}: A={materials['A']}, B={materials['B']}, C={materials['C']}")
        
        # Create arrangement tuple - using the specific channel assignments
        arrangement = (materials.get('A'), materials.get('B'), materials.get('C'))
        
        # Collect difference columns for this file
        file_diff_data = file_df[['datetime']].copy()
        file_has_diffs = False
        
        for col in file_df.columns:
            if '_minus_' in col:
                all_diff_columns.add(col)
                file_diff_data[col] = file_df[col]
                file_has_diffs = True
                
                # Also handle pair-specific grouping
                mat1, mat2 = col.split('_minus_')
                if mat1 in rev_materials and mat2 in rev_materials:
                    c1, c2 = rev_materials[mat1], rev_materials[mat2]
                    if c1 > c2:
                        c1, c2 = c2, c1
                    pair_key = c1 + c2
                    if pair_key in pair_arrangement_data:
                        pair_mats = (materials[c1], materials[c2])

                        if pair_mats not in pair_arrangement_data[pair_key]:
                            pair_arrangement_data[pair_key][pair_mats] = []

                        pair_df = file_df[['datetime', col]].copy()
                        pair_arrangement_data[pair_key][pair_mats].append(pair_df)
        
        if file_has_diffs:
            if arrangement not in arrangement_data:
                arrangement_data[arrangement] = []
            arrangement_data[arrangement].append(file_diff_data)

    print(f"Found {len(arrangement_data)} unique material arrangements")
    print(f"Found difference columns: {list(all_diff_columns)}")
    
    # Process each arrangement group
    all_arrangement_averages = []
    
    for arrangement, df_list in arrangement_data.items():
        print(f"Processing arrangement {arrangement} with {len(df_list)} files...")
        
        # Wrap timestamps and snap to minute for each df in this arrangement
        wrapped_dfs = []
        for df in df_list:
            df = df.copy()
            df['datetime'] = df['datetime'].apply(wrap_datetime_to_time)
            df['datetime'] = df['datetime'].dt.floor('T')
            df = df.groupby('datetime').mean().reset_index()
            wrapped_dfs.append(df)
        
        # Combine all data for this arrangement
        arrangement_combined = pd.concat(wrapped_dfs, ignore_index=True)
        
        # Average across all instances of this arrangement
        agg_dict = {col: 'mean' for col in all_diff_columns if col in arrangement_combined.columns}
        if agg_dict:
            arrangement_avg = arrangement_combined.groupby('datetime').agg(agg_dict).reset_index()
            all_arrangement_averages.append(arrangement_avg)
            print(f"  Arrangement {arrangement} has {len(arrangement_avg)} averaged timesteps")
    
    if not all_arrangement_averages:
        print("No arrangements with difference columns found")
        return None, {}, {}
    
    # Combine all arrangement averages
    all_averages = pd.concat(all_arrangement_averages, ignore_index=True)
    final_agg_dict = {col: 'mean' for col in all_diff_columns if col in all_averages.columns}
    final_avg = all_averages.groupby('datetime').agg(final_agg_dict).reset_index()
    
    print(f"Final averaged data has {len(final_avg)} timesteps")
    
    avg_diffs = {col: final_avg[col] for col in all_diff_columns if col in final_avg.columns}

    # Process pair-specific averages
    pair_avg_diffs = {}
    for pair, pair_dict in pair_arrangement_data.items():
        if not pair_dict:
            continue
            
        print(f"Processing pair {pair} with {len(pair_dict)} unique material combinations...")
        
        # Process each unique material combination for this pair
        pair_all_avgs = []
        for pair_mats, df_list in pair_dict.items():
            # Wrap and average within this material combination
            wrapped_dfs = []
            for df in df_list:
                df = df.copy()
                df['datetime'] = pd.to_datetime(df['datetime']).apply(wrap_datetime_to_time)
                df['datetime'] = df['datetime'].dt.floor('T')
                df = df.groupby('datetime').mean().reset_index()
                wrapped_dfs.append(df)
            
            # Combine and average for this material combination
            if wrapped_dfs:
                combined = pd.concat(wrapped_dfs, ignore_index=True)
                agg_dict = {col: 'mean' for col in combined.columns if col not in ['datetime']}
                avg = combined.groupby('datetime').agg(agg_dict).reset_index()
                pair_all_avgs.append(avg)
        
        # Combine all material combinations for this pair
        if pair_all_avgs:
            pair_combined = pd.concat(pair_all_avgs, ignore_index=True)
            agg_dict = {col: 'mean' for col in pair_combined.columns if col not in ['datetime']}
            pair_final_avg = pair_combined.groupby('datetime').agg(agg_dict).reset_index()
            pair_avg_diffs[pair] = pair_final_avg

    return final_avg['datetime'], avg_diffs, pair_avg_diffs


def plot_material_channels(df, fig, colors, wrap, opacity):
    """Plot temperature channels grouped by material"""
    # Collect material/channel combinations
    all_material_channels = {}
    for col in df.columns:
        if '/' in col and col.split('/')[1] in ['A', 'B', 'C']:
            material, channel = col.split('/')
            if material not in all_material_channels:
                all_material_channels[material] = []
            all_material_channels[material].append(channel)
    
    # Plot by material
    for material, channels_list in all_material_channels.items():
        unique_channels = list(set(channels_list))
        
        if len(unique_channels) == 1:
            # Single channel - use Material/Channel format
            channel = unique_channels[0]
            material_cols = [col for col in df.columns if col == f'{material}/{channel}']
            combined_data = pd.concat([df[col] for col in material_cols], axis=0)
            combined_data = combined_data.groupby(combined_data.index).first().sort_index()
            
            label = f'{material}/{channel}'
            color = colors[material]
            
            channel_traces = plot_trace_with_gaps(
                wrap=wrap,
                opacity=opacity,
                x=df['datetime'],
                y=combined_data,
                mode='lines',
                name=label,
                line=dict(color=color, width=3),
                hovertemplate=f'{label}: %{{y:.1f}}°C<br>%{{x}}<extra></extra>',
                legendgroup=material
            )
            for trace in channel_traces:
                fig.add_trace(trace, secondary_y=False, row=1, col=1)
        else:
            # Multiple channels - use Material/X format
            label = f'{material}/X'
            color = colors[material]
            
            first_trace = True
            for channel in unique_channels:
                material_cols = [col for col in df.columns if col == f'{material}/{channel}']
                combined_data = pd.concat([df[col] for col in material_cols], axis=0)
                combined_data = combined_data.groupby(combined_data.index).first().sort_index()
                
                show_legend = first_trace
                trace_name = label if first_trace else None
                
                channel_traces = plot_trace_with_gaps(
                    wrap=wrap,
                    opacity=opacity,
                    x=df['datetime'],
                    y=combined_data,
                    mode='lines',
                    name=trace_name,
                    line=dict(color=color, width=3),
                    hovertemplate=f'{material}/{channel}: %{{y:.1f}}°C<br>%{{x}}<extra></extra>',
                    showlegend=show_legend,
                    legendgroup=material
                )
                for trace in channel_traces:
                    fig.add_trace(trace, secondary_y=False, row=1, col=1)
                first_trace = False
    
    # Handle plain channel columns
    for channel in ['A', 'B', 'C']:
        if channel in df.columns:
            label = channel
            color = colors[f'{channel}_default']
            
            channel_traces = plot_trace_with_gaps(
                wrap=wrap,
                opacity=opacity,
                x=df['datetime'],
                y=df[channel],
                mode='lines',
                name=label,
                line=dict(color=color, width=3),
                hovertemplate=f'{label}: %{{y:.1f}}°C<br>%{{x}}<extra></extra>'
            )
            for trace in channel_traces:
                fig.add_trace(trace, secondary_y=False, row=1, col=1)


def plot_radiation_data(df, fig, colors, wrap, opacity):
    """Plot radiation variables on secondary y-axis"""
    radiation_vars = ['solar', 'ir_sky', 'ir_net', 'ir_in']
    for var in radiation_vars:
        if var in df.columns:
            radiation_traces = plot_trace_with_gaps(
                wrap=wrap,
                opacity=opacity,
                x=df['datetime'],
                y=df[var],
                mode='lines',
                name=var,
                line=dict(color=colors[var], width=3, dash='dash'),
                hovertemplate=f'{var}: %{{y:.1f}} W/m²<br>%{{x}}<extra></extra>'
            )
            for trace in radiation_traces:
                fig.add_trace(trace, secondary_y=True, row=1, col=1)


def plot_temperature_differences(df, fig, wrap, opacity, lw):
    """Plot all temperature difference traces"""
    _plot_temperature_differences_impl(df, fig, wrap, opacity, lw, row=2)


def plot_temperature_differences_single(df, fig, wrap, opacity, lw, short_names=False, legend_name='legend2'):
    """Plot all temperature difference traces for a single-subplot figure"""
    _plot_temperature_differences_impl(df, fig, wrap, opacity, lw, row=1, short_names=short_names, legend_name=legend_name)


def _plot_temperature_differences_impl(df, fig, wrap, opacity, lw, row, short_names=False, legend_name='legend2'):
    """Implementation for plotting temperature difference traces"""
    # Order traces with Boro-CaF2 and Sapph-CaF2 first
    diff_configs = [
        ('Boro_minus_CaF2', 'Boro', 'CaF2', 'Boro'),
        ('Sapph_minus_CaF2', 'Sapph', 'CaF2', 'Sapph'),
        ('Boro_minus_Empty', 'Boro', 'Empty', 'Empty'),
        ('Aluminum_minus_Acrylic', 'Aluminum', 'Acrylic', 'Acrylic'),
        ('Aluminum_minus_air', 'Aluminum', 'air', 'air'),
        ('AlumAlum_minus_BlackBlack', 'AlumAlum', 'BlackBlack', 'AlumAlum'),
        ('AlumAlum_minus_Alt', 'AlumAlum', 'Alt', 'CaF2'),
        ('BlackBlack_minus_Alt', 'BlackBlack', 'Alt', 'BlackBlack'),
        ('air_minus_BlackBlack', 'air', 'BlackBlack', 'air'),
        ('A_minus_B', 'A', 'B', 'A_default'),
        ('A_minus_C', 'A', 'C', 'C_default'),
        ('B_minus_C', 'B', 'C', 'B_default'),
    ]

    for diff_col, mat1, mat2, color_key in diff_configs:
        plot_temperature_difference(df, diff_col, mat1, mat2, color_key, fig, wrap=wrap, opacity=opacity, lw=lw, row=row, short_names=short_names, legend_name=legend_name)


def plot_channel_pair_differences(df, fig, wrap, opacity, lw, channel_pair, row, file_materials):
    """Plot material-based differences for a specific channel pair"""
    colors = graph_config.TRACE_COLORS
    ch1, ch2 = channel_pair
    used_files = []
    
    for source_file, materials in file_materials.items():
        # always put CaF2 as 2nd material
        if materials[ch1] == 'CaF2':
            ch1, ch2 = ch2, ch1

        if materials[ch1] and materials[ch2] and materials[ch1] != materials[ch2]:
            diff_col = f'{materials[ch1]}_minus_{materials[ch2]}'
            if diff_col not in df.columns:
                # check other way
                diff_col = f'{materials[ch2]}_minus_{materials[ch1]}'

            if diff_col in df.columns:
                print("From %s, plotting %s vs %s from %s on %s" % (
                    source_file, materials[ch1], materials[ch2], diff_col, channel_pair
                ))
                # Filter data for this specific file
                file_mask = df['source_file'] == source_file if 'source_file' in df.columns else pd.Series([True]*len(df))
                file_df = df[file_mask]
                
                if not file_df.empty and file_df[diff_col].notna().any():
                    used_files.append(source_file)
                    traces = plot_trace_with_gaps(
                        wrap=wrap,
                        opacity=opacity,
                        x=file_df['datetime'],
                        y=file_df[diff_col],
                        mode='lines',
                        name=f'{materials[ch1]} - {materials[ch2]}',
                        line=dict(color=colors.get(materials[ch1], '#808080'), width=lw),
                        hovertemplate=f'{materials[ch1]} - {materials[ch2]}: %{{y:.2f}}°C<br>%{{x}}<extra></extra>',
                        legendgroup=f'{ch1}&{ch2}_{materials[ch1]}_{materials[ch2]}'
                    )
                    for trace in traces:
                        trace.legend = f'legend{row}'
                        fig.add_trace(trace, row=row, col=1)
    
    return used_files


def plot_averaged_traces(avg_datetime, avg_diffs, fig, colors, wrap, opacity_avg, lw_avg, row=2, short_names=False):
    """Plot averaged temperature difference traces"""
    # Order of averaged traces to match regular traces
    diff_order = [
        'Boro_minus_CaF2',
        'Sapph_minus_CaF2',
        'Boro_minus_Empty',
        'Aluminum_minus_Acrylic',
        'Aluminum_minus_air',
        'A_minus_B',
        'A_minus_C',
        'B_minus_C'
    ]

    if short_names:
        diff_mappings = {
            'Boro_minus_CaF2': ('Boro (Avg)', 'Boro'),
            'Sapph_minus_CaF2': ('Sapph (Avg)', 'Sapph'),
            'Aluminum_minus_Acrylic': ('Aluminum (Avg)', 'Acrylic'),
            'Aluminum_minus_air': ('Aluminum (Avg)', 'air'),
            'A_minus_B': ('A (Avg)', 'A_default'),
            'A_minus_C': ('A (Avg)', 'A_default'),
            'B_minus_C': ('B (Avg)', 'B_default'),
            'Boro_minus_Empty': ('Boro (Avg)', 'Empty')
        }
    else:
        diff_mappings = {
            'Boro_minus_CaF2': ('ΔT: Boro vs. CaF2 (Average)', 'Boro'),
            'Sapph_minus_CaF2': ('ΔT: Sapph vs. CaF2 (Average)', 'Sapph'),
            'Aluminum_minus_Acrylic': ('ΔT: Aluminum vs. Acrylic (Average)', 'Acrylic'),
            'Aluminum_minus_air': ('ΔT: Aluminum vs. Air Temp (Average)', 'air'),
            'A_minus_B': ('ΔT: A vs. B (Average)', 'A_default'),
            'A_minus_C': ('ΔT: A vs. C (Average)', 'A_default'),
            'B_minus_C': ('ΔT: B vs. C (Average)', 'B_default'),
            'Boro_minus_Empty': ('ΔT: Boro vs. Empty (Average)', 'Empty')
        }
        if os.environ.get("TWOPANE"):
            diff_mappings.update({
                'Boro_minus_CaF2': ('ΔT: 2xBoro vs. Boro/CaF2 (Average)', 'Boro'),
                'Boro_minus_Empty': ('ΔT: 2xBoro vs. 1xBoro (Average)', 'Empty')
            })

        # Plot in specified order
    for diff_col in diff_order:
        if diff_col not in avg_diffs or diff_col not in diff_mappings:
            continue
            
        avg_data = avg_diffs[diff_col]
        name, color_key = diff_mappings[diff_col]
        color = colors.get(color_key, '#000000')
        
        # Determine if this should go on secondary y-axis
        use_secondary_y = (DUAL_YAXIS_ALAC and diff_col == 'Aluminum_minus_Acrylic')
        
        # Black backing trace
        black_traces = plot_trace_with_gaps(
            wrap=wrap,
            opacity=opacity_avg,
            x=avg_datetime,
            y=avg_data,
            mode='lines',
            name=name,
            legendgroup=f'delta_{color_key}_avg',
            line=dict(color='black', width=lw_avg+4),
            hovertemplate=f'{name}: %{{y:.1f}}°C<br>%{{x}}<extra></extra>',
            showlegend=False
        )
        for trace in black_traces:
            trace.legend = 'legend2'
            fig.add_trace(trace, row=row, col=1, secondary_y=use_secondary_y)

        # Colored average trace
        avg_traces = plot_trace_with_gaps(
            wrap=wrap,
            opacity=opacity_avg,
            x=avg_datetime,
            y=avg_data,
            mode='lines',
            name=name,
            legendgroup=f'delta_{color_key}_avg',
            line=dict(color=color, width=lw_avg),
            hovertemplate=f'{name}: %{{y:.1f}}°C<br>%{{x}}<extra></extra>'
        )
        for trace in avg_traces:
            trace.legend = 'legend2'
            fig.add_trace(trace, row=row, col=1, secondary_y=use_secondary_y)


def plot_channel_pair_averaged_traces(avg_datetime, channel_pair_avgs, fig, colors, wrap, opacity_avg, lw_avg, channel_pair, row):
    """Plot averaged traces for a specific channel pair"""
    pair_avgs = channel_pair_avgs.get(channel_pair, {})
    
    for diff_col, avg_data in pair_avgs.items():
        parts = diff_col.split('_minus_')
        if len(parts) == 2:
            mat1, mat2 = parts
            name = f'{mat1} - {mat2} (Avg)'
            color = colors.get(mat1, '#808080')
            
            # Black backing trace
            black_traces = plot_trace_with_gaps(
                wrap=wrap,
                opacity=opacity_avg,
                x=avg_datetime,
                y=avg_data,
                mode='lines',
                name=name,
                legendgroup=f'{channel_pair}_{diff_col}_avg',
                line=dict(color='black', width=lw_avg+4),
                hovertemplate=f'{name}: %{{y:.1f}}°C<br>%{{x}}<extra></extra>',
                showlegend=False
            )
            for trace in black_traces:
                trace.legend = f'legend{row}'
                fig.add_trace(trace, row=row, col=1)
            
            # Colored average trace
            avg_traces = plot_trace_with_gaps(
                wrap=wrap,
                opacity=opacity_avg,
                x=avg_datetime,
                y=avg_data,
                mode='lines',
                name=name,
                legendgroup=f'{channel_pair}_{diff_col}_avg',
                line=dict(color=color, width=lw_avg),
                hovertemplate=f'{name}: %{{y:.1f}}°C<br>%{{x}}<extra></extra>'
            )
            for trace in avg_traces:
                trace.legend = f'legend{row}'
                fig.add_trace(trace, row=row, col=1)


def _get_enhanced_html_style():
    """Return common HTML styling for enhanced plots"""
    return '''<head>
            <style>
                body { 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    margin: 0;
                    padding: 20px;
                    font-family: 'Arial', sans-serif;
                }
                .plotly-graph-div {
                    border-radius: 15px;
                    box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                    background: white;
                    padding: 10px;
                }
            </style>'''


def plot_processed_data(df, wrap=False, opacity=1.0, no_se=False, no_v=False, average=False, opacity_avg=1.0, lw=3, lw_avg=3, all_csv_files=None,
                       title_1=None, title_2=None, title_3=None, title_4=None, title_5=None, title_main=None):
    """Create interactive dual-axis plot of processed roofalum data"""
    
    # Find data bounds
    columns = [col for col in df.columns if 'datetime' not in col]
    has_data_mask = df[columns].notna().any(axis=1)
    data_start_time = df.loc[has_data_mask, 'datetime'].min()
    data_end_time = df.loc[has_data_mask, 'datetime'].max()

    # Load sun data
    sun_data = load_sun_data()
    data_dates = df.loc[has_data_mask, 'datetime'].dt.date.unique()

    # Store original df for condition filtering
    original_df = df.copy()
    
    # Transform data for wrapping if requested
    if wrap:
        df = df.copy()
        df['datetime'] = df['datetime'].apply(wrap_datetime_to_time)

    # Set up colors
    colors = graph_config.TRACE_COLORS

    # Create subplots
    subplot_titles = (
        title_1 or "Temperature and Radiation",
        title_2 or "Temperature Differences", 
        title_3 or "A&B Differences",
        title_4 or "A&C Differences",
        title_5 or "B&C Differences"
    )
    
    # Configure secondary_y for row 2 if dual y-axis mode is enabled
    row2_secondary_y = DUAL_YAXIS_ALAC
    
    fig = make_subplots(
        rows=5, cols=1,
        specs=[[{"secondary_y": True}], [{"secondary_y": row2_secondary_y}], [{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}]],
        subplot_titles=subplot_titles,
        vertical_spacing=0.05,
        shared_xaxes=True,
        row_heights=graph_config.ROW_HEIGHTS,
    )

    # Get material assignments and condition ranges BEFORE wrapping
    file_condition_ranges = None
    row_files_map = {}
    
    if all_csv_files:
        file_materials = get_material_assignments_per_file(all_csv_files)
        file_condition_ranges = get_condition_ranges_per_file(all_csv_files)

    # Plot temperatures
    add_temperature_trace(df, 'air', 'air', colors['air'], fig, wrap=wrap, opacity=opacity, row=1)
    
    # Plot material channels
    plot_material_channels(df, fig, colors, wrap, opacity)
    
    # Plot pyrge temperature
    if 'pyrge' in df.columns:
        add_temperature_trace(df, 'pyrge', 'pyrge', colors['pyrge'], fig, wrap=wrap, opacity=opacity, row=1)

    # Plot radiation data
    plot_radiation_data(df, fig, colors, wrap, opacity)

    # Plot temperature differences
    plot_temperature_differences(df, fig, wrap, opacity, lw)
    
    # Plot channel pair differences and track which files are used
    if all_csv_files:
        # Plot differences for each channel pair and collect used files
        row_files_map[3] = plot_channel_pair_differences(df, fig, wrap, opacity, lw, ('A', 'B'), 3, file_materials)
        row_files_map[4] = plot_channel_pair_differences(df, fig, wrap, opacity, lw, ('A', 'C'), 4, file_materials)
        row_files_map[5] = plot_channel_pair_differences(df, fig, wrap, opacity, lw, ('B', 'C'), 5, file_materials)
    
    # Add condition backgrounds with filtering (skip with --no-v for fast iteration)
    if not no_v:
        add_condition_backgrounds(fig, original_df, wrap=wrap,
                                file_condition_ranges=file_condition_ranges,
                                row_files_map=row_files_map)
    
    # Calculate and plot averages if requested
    avg_diffs = None
    if average and wrap and all_csv_files:
        print("Calculating averaged temperature differences...")
        avg_datetime, avg_diffs, channel_pair_avgs = calculate_averaged_temp_diffs(all_csv_files)
        if avg_datetime is not None and avg_diffs:
            print("Adding averaged traces with bold styling...")
            plot_averaged_traces(avg_datetime, avg_diffs, fig, colors, wrap, opacity_avg, lw_avg)

            # Plot channel pair averages
            plot_channel_pair_averaged_traces(avg_datetime, channel_pair_avgs, fig, colors, wrap, opacity_avg, lw_avg, 'AB', 3)
            plot_channel_pair_averaged_traces(avg_datetime, channel_pair_avgs, fig, colors, wrap, opacity_avg, lw_avg, 'AC', 4)
            plot_channel_pair_averaged_traces(avg_datetime, channel_pair_avgs, fig, colors, wrap, opacity_avg, lw_avg, 'BC', 5)

        if avg_diffs:
            output_avg_diff_summary(avg_diffs)

    # Add reference lines
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5, secondary_y=False, row=1, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5, secondary_y=True, row=1, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5, row=2, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5, row=3, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5, row=4, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5, row=5, col=1)

    # Add data start/end markers
    if not no_se:
        start_plot = wrap_datetime_to_time(data_start_time) if wrap else data_start_time
        end_plot = wrap_datetime_to_time(data_end_time) if wrap else data_end_time
        add_vertical_lines_to_subplots(fig, start_plot, "Data Start", color="#0D4C0D", width=2, dash="dash")
        add_vertical_lines_to_subplots(fig, end_plot, "Data End", color="#540B0B", width=2, dash="dash")

    # Add solar noon lines (skip with --no-v for fast iteration)
    if not no_v:
        for date in data_dates:
            solar_noon_dt = get_solar_noon_datetime(date, sun_data)
            if solar_noon_dt is not None:
                if not wrap and solar_noon_dt < data_start_time:
                    continue
                solar_plot = wrap_datetime_to_time(solar_noon_dt) if wrap else solar_noon_dt
                add_vertical_lines_to_subplots(fig, solar_plot, "Solar Noon", color=colors['solar'], width=2, dash="dash")

    # Configure axes
    fig.update_yaxes(title_text="Temperature (°C)", title_font=dict(size=graph_config.AXISFONT_SIZE), 
                     tickfont=dict(size=graph_config.TICKFONT_SIZE), range=graph_config.TEMP_Y_RANGE, 
                     secondary_y=False, row=1, col=1)
    fig.update_yaxes(title_text="Radiation (W/m²)", title_font=dict(size=graph_config.AXISFONT_SIZE), 
                     tickfont=dict(size=graph_config.TICKFONT_SIZE), range=graph_config.RADIATION_Y_RANGE, 
                     secondary_y=True, row=1, col=1)
    
    # Configure y-axes for all difference subplots
    tick_vals = np.arange(-5, 50, graph_config.TEMP_DIFF_TICK_INTERVAL)

    # if have aluminum in data then use 1 tick interval
    if 'Aluminum_minus_Acrylic' in df.columns or 'Aluminum_minus_air' in df.columns:
        tick_vals = np.arange(-5, 50, graph_config.TEMP_DIFF_TICK_INTERVAL_ALUMINUM)

    for row in [2, 3, 4, 5]:
        # Use custom range for row 2 if dual y-axis mode is enabled
        if row == 2 and DUAL_YAXIS_ALAC:
            y_range = [-4, 12]
            y_title = "Al-Air ΔT (°C)"
        else:
            y_range = graph_config.TEMP_DIFF_Y_RANGE
            y_title = "ΔT (°C)"
            
        fig.update_yaxes(title_text=y_title, title_font=dict(size=graph_config.AXISFONT_SIZE-2), 
                         tickfont=dict(size=graph_config.TICKFONT_SIZE), range=y_range,
                         showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                         dtick=1, row=row, col=1, secondary_y=False)
        fig.update_yaxes(tickvals=tick_vals, ticktext=[f"{x:.1f}" for x in tick_vals],
                         tickfont=dict(size=graph_config.TICKFONT_SIZE), row=row, col=1, secondary_y=False)
    
    # Configure secondary y-axis for row 2 if dual y-axis mode is enabled
    if DUAL_YAXIS_ALAC:
        # Left axis: -4 to 12, zero at position 4/16 = 0.25 from bottom
        # Right axis: to align zeros, if max is 1.4, then min = -1.4 * (4/12) = -0.467
        right_min = -1.4 * (4.0/12.0)  # -0.467
        right_max = 1.4
        right_ticks = np.arange(-0.4, 1.5, 0.2)
        fig.update_yaxes(title_text="Al-Acrylic ΔT (°C)", title_font=dict(size=graph_config.AXISFONT_SIZE-2),
                         tickfont=dict(size=graph_config.TICKFONT_SIZE), range=[right_min, right_max],
                         tickvals=right_ticks, ticktext=[f"{x:.1f}" for x in right_ticks],
                         showgrid=False, row=2, col=1, secondary_y=True)

    # Configure x-axes
    for row in [1, 2, 3, 4, 5]:
        fig.update_xaxes(tickformat='%H:%M' if wrap else None, tickfont=dict(size=graph_config.TICKFONT_SIZE), 
                         mirror="allticks", showticklabels=True, row=row, col=1)

    # Update layout
    fig.update_layout(
        title={
            'text': title_main or 'Greenhouse Experiment',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': graph_config.TITLE_FONT_SIZE, 'color': 'black'}
        },
        xaxis_title="Time of Day (HH:MM)" if wrap else "Time (HH:MM)",
        xaxis_title_font=dict(size=graph_config.AXISFONT_SIZE),
        xaxis2_title="Time of Day (HH:MM)" if wrap else "Time (HH:MM)",
        xaxis2_title_font=dict(size=graph_config.AXISFONT_SIZE),
        xaxis3_title="Time of Day (HH:MM)" if wrap else "Time (HH:MM)",
        xaxis3_title_font=dict(size=graph_config.AXISFONT_SIZE),
        xaxis4_title="Time of Day (HH:MM)" if wrap else "Time (HH:MM)",
        xaxis4_title_font=dict(size=graph_config.AXISFONT_SIZE),
        xaxis5_title="Time of Day (HH:MM)" if wrap else "Time (HH:MM)",
        xaxis5_title_font=dict(size=graph_config.AXISFONT_SIZE),
        font=dict(family="Computer Modern, Times New Roman, serif", color="black"),
        plot_bgcolor='rgba(248,249,250,0.8)',
        paper_bgcolor='white',
        width=graph_config.PLOT_WIDTH,
        height=graph_config.PLOT_HEIGHT,
        hovermode='x unified',
        legend=dict(
            orientation="v",
            yanchor="top", y=1,
            xanchor="left", x=1.02,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
            font=dict(size=graph_config.LEGEND_FONT_SIZE),
            tracegroupgap=10
        ),
        legend2=dict(
            orientation="v",
            yanchor="top", y=0.6,
            xanchor="left", x=1.02,
            bgcolor="rgba(240,248,255,0.9)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
            font=dict(size=graph_config.LEGEND_FONT_SIZE),
        ),
        legend3=dict(
            orientation="v",
            yanchor="top", y=0.4,
            xanchor="left", x=1.02,
            bgcolor="rgba(255,240,245,0.9)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
            font=dict(size=graph_config.LEGEND_FONT_SIZE-1),
        ),
        legend4=dict(
            orientation="v",
            yanchor="top", y=0.25,
            xanchor="left", x=1.02,
            bgcolor="rgba(245,255,240,0.9)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
            font=dict(size=graph_config.LEGEND_FONT_SIZE-1),
        ),
        legend5=dict(
            orientation="v",
            yanchor="top", y=0.1,
            xanchor="left", x=1.02,
            bgcolor="rgba(255,245,240,0.9)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
            font=dict(size=graph_config.LEGEND_FONT_SIZE-1),
        ),
        dragmode='zoom',
        selectdirection='d'
    )

    # Configure grid and spikes
    grid_config = dict(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        showspikes=True,
        spikemode='across',
        spikesnap='cursor',
        spikecolor='rgba(0,0,0,0.3)',
        spikethickness=1,
        tickformat='%H:%M' if wrap else None,
        tickfont=dict(size=graph_config.TICKFONT_SIZE)
    )
    
    for row in [1, 2, 3, 4, 5]:
        fig.update_xaxes(grid_config, row=row, col=1)
    
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)', 
                     tickfont=dict(size=graph_config.TICKFONT_SIZE), row=1, col=1)
    fig.update_yaxes(showgrid=False, tickfont=dict(size=graph_config.TICKFONT_SIZE), 
                     secondary_y=True, row=1, col=1)

    fig.update_traces(line_shape='linear')
    fig.update_annotations(font_size=graph_config.SUBTITLE_FONT_SIZE)

    # Create HTML file and open
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
        html_content = fig.to_html(
            include_plotlyjs='cdn',
            config={
                'displayModeBar': True,
                'scrollZoom': True,
                'doubleClick': 'reset+autosize',
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': f'roofalum_graph',
                    'height': graph_config.PLOT_HEIGHT,
                    'width': graph_config.PLOT_WIDTH,
                    'scale': 2
                },
            }
        )

        enhanced_html = html_content.replace('<head>', _get_enhanced_html_style())
        f.write(enhanced_html)
        temp_file = f.name

    # show avg at end too
    if avg_diffs:
        output_avg_diff_summary(avg_diffs)

    webbrowser.open(f'file://{temp_file}')
    return temp_file


def output_avg_diff_summary(avg_diffs):
    ranges = {
        'peak': [((12, 00), (15, 00))],
        'direct_sun': [((8, 30), (18, 30))],
        'indirect_sun': [((6, 0), (8, 30)), ((18, 30), (21, 0))],
        'night': [((0, 0), (6, 0)), ((21, 0), (24, 0))]
    }

    headers = ['material', 'overall', 'peak', 'direct_sun', 'indirect_sun', 'night']
    rows = []
    for col in ['Boro_minus_CaF2', 'Sapph_minus_CaF2']:
        if col not in avg_diffs:
            continue

        coldf = avg_diffs[col]
        mat = col.split("_")[0]
        row = [
            mat,
            "%.2f $\pm$ %.2f" % (
                coldf.mean(), coldf.std(),
            ),
        ]
        for range_name in headers[2:]:
            rangedf = pd.concat([coldf[sh*60+sm:eh*60+em] for ((sh, sm), (eh, em)) in ranges[range_name]])
            row.append("%.2f $\pm$ %.2f" % (
                rangedf.mean(), rangedf.std(),
            ))
        rows.append(row)

    print(tabulate.tabulate(rows, headers=headers, tablefmt='grid'))
    # import code; code.interact(local={**locals(), **globals()})


def plot_paper_temp_diffs(df, wrap=False, opacity=1.0, no_v=False, average=False, opacity_avg=1.0, lw=3, lw_avg=3, all_csv_files=None,
                          title=None, output=None):
    """Create a single Temperature Differences plot for paper figures

    Args:
        output: Output filename (without path). Defaults to 'clearsky_allcombo_averages.pdf'
    """

    # Find data bounds
    columns = [col for col in df.columns if 'datetime' not in col]
    has_data_mask = df[columns].notna().any(axis=1)
    data_start_time = df.loc[has_data_mask, 'datetime'].min()
    data_end_time = df.loc[has_data_mask, 'datetime'].max()

    # Load sun data
    sun_data = load_sun_data()
    data_dates = df.loc[has_data_mask, 'datetime'].dt.date.unique()

    # Store original df for condition filtering
    original_df = df.copy()

    # Transform data for wrapping if requested
    if wrap:
        df = df.copy()
        # Convert datetime to hours (0-24) for clean numeric x-axis in PDF
        df['datetime'] = df['datetime'].dt.hour + df['datetime'].dt.minute / 60 + df['datetime'].dt.second / 3600

    # Set up colors
    colors = graph_config.TRACE_COLORS

    # Create single-row subplot figure (needed for compatibility with trace-adding functions)
    fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": DUAL_YAXIS_ALAC}]])

    # Add condition backgrounds (skip with --no-v for fast iteration)
    if not no_v:
        add_condition_backgrounds(fig, original_df, wrap=wrap, numeric_x=wrap, rows=(1,))

    # Plot temperature differences (row=1 since this is single subplot)
    # Use 'legend' for raw traces (left column of 2x2 legend)
    plot_temperature_differences_single(df, fig, wrap, opacity, lw, short_names=True, legend_name='legend')

    # Calculate and plot averages if requested
    avg_diffs = None
    if average and wrap and all_csv_files:
        print("Calculating averaged temperature differences...")
        avg_datetime, avg_diffs, channel_pair_avgs = calculate_averaged_temp_diffs(all_csv_files)
        if avg_datetime is not None and avg_diffs:
            # Convert avg_datetime to hours for numeric x-axis
            avg_hours = avg_datetime.dt.hour + avg_datetime.dt.minute / 60 + avg_datetime.dt.second / 3600
            print("Adding averaged traces with bold styling...")
            plot_averaged_traces(avg_hours, avg_diffs, fig, colors, wrap, opacity_avg, lw_avg, row=1, short_names=True)

        if avg_diffs:
            output_avg_diff_summary(avg_diffs)

    # Add reference line at y=0
    fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=2)

    # Add solar noon lines (skip with --no-v for fast iteration)
    if not no_v:
        for date in data_dates:
            solar_noon_dt = get_solar_noon_datetime(date, sun_data)
            if solar_noon_dt is not None:
                if not wrap and solar_noon_dt < data_start_time:
                    continue
                if wrap:
                    # Convert to numeric hours for numeric x-axis
                    solar_plot = solar_noon_dt.hour + solar_noon_dt.minute / 60 + solar_noon_dt.second / 3600
                else:
                    solar_plot = solar_noon_dt
                fig.add_vline(x=solar_plot, line_dash="dash", line_color=colors['solar'], line_width=2)

    # Configure y-axis
    tick_vals = np.arange(-5, 50, graph_config.TEMP_DIFF_TICK_INTERVAL)
    if 'Aluminum_minus_Acrylic' in df.columns or 'Aluminum_minus_air' in df.columns:
        tick_vals = np.arange(-5, 50, graph_config.TEMP_DIFF_TICK_INTERVAL_ALUMINUM)

    fig.update_yaxes(
        title_text="ΔT vs. CaF₂ (°C)",
        title_font=dict(size=42),
        tickfont=dict(size=39),
        range=graph_config.TEMP_DIFF_Y_RANGE,
        tickvals=tick_vals,
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(200,200,200,0.5)',
    )

    # Configure x-axis
    xaxis_config = dict(
        title_text="Time of Day (HH:MM)" if wrap else "Time (HH:MM)",
        title_font=dict(size=42),
        tickfont=dict(size=39),
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(200,200,200,0.5)',
    )
    if wrap:
        # Simple numeric ticks every 3 hours
        xaxis_config['tickmode'] = 'array'
        xaxis_config['tickvals'] = [0, 3, 6, 9, 12, 15, 18, 21, 24]
        xaxis_config['ticktext'] = ['00:00', '03:00', '06:00', '09:00', '12:00', '15:00', '18:00', '21:00', '00:00']
        xaxis_config['range'] = [0, 24]
    fig.update_xaxes(**xaxis_config)

    # Update layout (matching pane_tra transmission style)
    fig.update_layout(
        font=dict(family="Times New Roman, Times, serif", color="black"),
        plot_bgcolor='#f5f5f5',
        paper_bgcolor='white',
        width=1200,
        height=520,
        hovermode='x unified',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=0.0,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(0,0,0,0)",
            borderwidth=0,
            font=dict(size=35),
            title=dict(text="", font=dict(size=1)),
        ),
        legend2=dict(
            orientation="v",
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=0.12,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(0,0,0,0)",
            borderwidth=0,
            font=dict(size=35),
            title=dict(text="", font=dict(size=1)),
        ),
        margin=dict(l=100, r=20, t=40, b=130),
    )

    fig.update_traces(line_shape='linear', connectgaps=False)

    # Output PDF to figures directory
    figures_dir = Path(__file__).parent / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    output_filename = output if output else 'clearsky_allcombo_averages.pdf'
    pdf_file = figures_dir / output_filename
    fig.write_image(str(pdf_file), format='pdf', width=graph_config.PLOT_WIDTH, height=600)
    print(f"PDF saved: {pdf_file}")

    # show avg at end too
    if avg_diffs:
        output_avg_diff_summary(avg_diffs)

    return str(pdf_file)


class CmdLine:
    def _align_ir_to_temp_timestamps(self, df):
        """Align IR timestamps to temperature timestamps using linear progression tracking"""
        print("Starting IR timestamp alignment...")
        
        df_aligned = df.copy()
        
        ir_columns = ['solar', 'ir_net', 'thermistorV', 'excitationV']
        temp_columns = ['air', 'A', 'B', 'C']
        
        has_temp_mask = df_aligned[temp_columns].notna().any(axis=1)
        has_ir_mask = df_aligned[ir_columns].notna().any(axis=1)
        
        last_temp_time = None
        last_temp_idx = None
        alignments_made = 0
        rows_to_drop = []
        
        for idx in df_aligned.index:
            current_time = df_aligned.loc[idx, 'datetime']
            has_temp_data = has_temp_mask.loc[idx]
            
            if has_temp_data:
                last_temp_time = current_time
                last_temp_idx = idx
                continue
            
            has_ir_data = has_ir_mask.loc[idx]
            
            if has_ir_data and last_temp_time is not None:
                time_diff = abs(current_time - last_temp_time)
                
                if time_diff <= pd.Timedelta(seconds=graph_config.MAX_TIME_DIFF_SECONDS):
                    for col in ir_columns:
                        df_aligned.loc[last_temp_idx, col] = df_aligned.loc[idx, col]
                    rows_to_drop.append(idx)
                    alignments_made += 1

        if rows_to_drop:
            df_aligned.drop(rows_to_drop, inplace=True)
        
        df_aligned = df_aligned.sort_values('datetime').reset_index(drop=True)
        print(f"Made {alignments_made} IR timestamp alignments")
        
        return df_aligned

    def _calculate_surface_differences(self, df):
        """Calculate temperature differences between surfaces"""
        df_with_diffs = df.copy()
        
        caf2_data = None
        boro_data = None
        other_materials = {}
        
        for col in df_with_diffs.columns:
            if col.startswith('CaF2/'):
                caf2_data = df_with_diffs[col]
            elif '/' in col and any(col.startswith(f'{mat}/') for mat in ['Boro', 'Sapph', 'Empty', 'Aluminum', 'Acrylic', 'AlumAlum', 'BlackBlack', 'Alt']):
                material = col.split('/')[0]
                other_materials[material] = df_with_diffs[col]

        print(f"Calculating surface differences...")
        
        if caf2_data is not None:
            for material, material_data in other_materials.items():
                diff_col_name = f'{material}_minus_CaF2'
                df_with_diffs[diff_col_name] = material_data - caf2_data
                print(f"  Added column: {diff_col_name}")

        # Special comparisons
        if 'Aluminum' in other_materials:
            if 'Acrylic' in other_materials:
                df_with_diffs['Aluminum_minus_Acrylic'] = other_materials['Aluminum'] - other_materials['Acrylic']
                print(f"  Added column: Aluminum_minus_Acrylic")
            
            if 'air' in df_with_diffs.columns:
                df_with_diffs['Aluminum_minus_air'] = other_materials['Aluminum'] - df_with_diffs['air']
                print(f"  Added column: Aluminum_minus_air")

        if 'Boro' in other_materials and 'Empty' in other_materials:
            df_with_diffs['Boro_minus_Empty'] = other_materials['Boro'] - other_materials['Empty']
            print(f"  Added column: Boro_minus_Empty")
        
        # New material comparisons
        if 'AlumAlum' in other_materials:
            if 'BlackBlack' in other_materials:
                df_with_diffs['AlumAlum_minus_BlackBlack'] = other_materials['AlumAlum'] - other_materials['BlackBlack']
                print(f"  Added column: AlumAlum_minus_BlackBlack")
            if 'Alt' in other_materials:
                df_with_diffs['AlumAlum_minus_Alt'] = other_materials['AlumAlum'] - other_materials['Alt']
                print(f"  Added column: AlumAlum_minus_Alt")
        
        if 'BlackBlack' in other_materials:
            if 'Alt' in other_materials:
                df_with_diffs['BlackBlack_minus_Alt'] = other_materials['BlackBlack'] - other_materials['Alt']
                print(f"  Added column: BlackBlack_minus_Alt")
            if 'air' in df_with_diffs.columns:
                df_with_diffs['air_minus_BlackBlack'] = df_with_diffs['air'] - other_materials['BlackBlack']
                print(f"  Added column: air_minus_BlackBlack")
        
        # Plain channel differences
        if all(ch in df_with_diffs.columns for ch in ['A', 'B', 'C']):
            df_with_diffs['A_minus_B'] = df_with_diffs['A'] - df_with_diffs['B']
            df_with_diffs['A_minus_C'] = df_with_diffs['A'] - df_with_diffs['C']
            df_with_diffs['B_minus_C'] = df_with_diffs['B'] - df_with_diffs['C']
            print(f"  Added columns: A_minus_B, A_minus_C, B_minus_C")
        
        return df_with_diffs

    def _apply_condition_ranges(self, df):
        """Apply condition ranges and store as binary columns"""
        df_with_conditions = df.copy()
        
        unique_conditions = list(set(range_data['condition'] for range_data in CONDITION_RANGES))
        condition_columns = {condition: condition.replace(' ', '_') for condition in unique_conditions}
        
        for condition, col_name in condition_columns.items():
            df_with_conditions[col_name] = 0
        
        print(f"Applying condition ranges...")
        
        for range_data in CONDITION_RANGES:
            start_time = pd.to_datetime(range_data['start'])
            end_time = pd.to_datetime(range_data['end'])
            condition = range_data['condition']
            col_name = condition_columns[condition]
            
            time_mask = (df_with_conditions['datetime'] >= start_time) & (df_with_conditions['datetime'] <= end_time)
            if time_mask.sum() > 0:
                df_with_conditions.loc[time_mask, col_name] = 1
        
        return df_with_conditions

    def process_data(self, csv_file, material_a=None, material_b=None, material_c=None):
        """Process raw CSV data and save to processed CSV"""
        if not Path(csv_file).exists():
            print(f"Error: File '{csv_file}' not found.")
            sys.exit(1)

        try:
            df = read_roofalum_csv(csv_file)
            df = self._align_ir_to_temp_timestamps(df)
            df = pre_filter_data(df)
            df = calculate_derived_values(df)

            # Rename channels with materials
            if material_a:
                df = df.rename(columns={'A': f'{material_a}/A'})
            if material_b:
                df = df.rename(columns={'B': f'{material_b}/B'})
            if material_c:
                df = df.rename(columns={'C': f'{material_c}/C'})

            df = self._calculate_surface_differences(df)
            df = self._apply_condition_ranges(df)

            # Save processed data
            script_dir = Path(__file__).parent
            output_dir = script_dir / 'data/main_exp/processed'
            output_dir.mkdir(parents=True, exist_ok=True)
            
            input_filename = Path(csv_file).stem
            output_file = output_dir / f'{input_filename}.csv'
            
            df.to_csv(output_file, index=False)
            print(f"Processed data saved to: {output_file}")

        except Exception as e:
            print(f"Error processing data: {e}")
            sys.exit(1)

    def plot1(self, *csv_files, wrap=False, opacity=1.0, no_se=False, no_v=False, average=False, opacity_avg=1.0, lw=3, lw_avg=3,
             title_1=None, title_2=None, title_3=None, title_4=None, title_5=None, title_main=None):
        """Generate interactive plot from processed CSV files"""
        if not csv_files:
            print("Error: At least one CSV file is required.")
            sys.exit(1)
        
        for csv_file in csv_files:
            if not Path(csv_file).exists():
                print(f"Error: File '{csv_file}' not found.")
                sys.exit(1)

        try:
            # Load and combine data
            all_data = []
            for csv_file in csv_files:
                print(f"Loading {csv_file}...")
                df = pd.read_csv(csv_file)
                df['datetime'] = pd.to_datetime(df['datetime'])
                df['source_file'] = csv_file
                all_data.append(df)
            
            df = pd.concat(all_data, ignore_index=True)

            # Generate plot
            plot_processed_data(
                df,
                wrap=wrap, opacity=opacity, no_se=no_se, no_v=no_v,
                average=average, opacity_avg=opacity_avg, lw=lw, lw_avg=lw_avg, all_csv_files=csv_files,
                title_1=title_1, title_2=title_2, title_3=title_3, title_4=title_4, title_5=title_5, title_main=title_main,
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error plotting data: {e}")
            sys.exit(1)

    def plot_paper_figures(self, *csv_files, wrap=False, opacity=1.0, no_v=False, average=False, opacity_avg=1.0, lw=3, lw_avg=3,
                          title=None, output=None):
        """Generate a single Temperature Differences plot for paper figures

        Args:
            output: Output filename (without path). Defaults to 'clearsky_allcombo_averages.pdf'
        """
        if not csv_files:
            print("Error: At least one CSV file is required.")
            sys.exit(1)

        for csv_file in csv_files:
            if not Path(csv_file).exists():
                print(f"Error: File '{csv_file}' not found.")
                sys.exit(1)

        try:
            # Load and combine data
            all_data = []
            for csv_file in csv_files:
                print(f"Loading {csv_file}...")
                df = pd.read_csv(csv_file)
                df['datetime'] = pd.to_datetime(df['datetime'])
                df['source_file'] = csv_file
                all_data.append(df)

            df = pd.concat(all_data, ignore_index=True)

            # Generate plot
            plot_paper_temp_diffs(
                df,
                wrap=wrap, opacity=opacity, no_v=no_v,
                average=average, opacity_avg=opacity_avg, lw=lw, lw_avg=lw_avg, all_csv_files=csv_files,
                title=title, output=output,
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error plotting data: {e}")
            sys.exit(1)


if __name__ == '__main__':
    fire.Fire(CmdLine)