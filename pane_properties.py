import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
import fire
import webbrowser
import tempfile
from scipy.interpolate import interp1d
from tabulate import tabulate


# Material colors from graph_config
MATERIAL_COLORS = {
    'Boro': '#ff7e00',
    'CaF2': '#00b7ef',
    'Sapph': '#6f3198',
    'Saph': '#6f3198',
}

# Full material names for figures
MATERIAL_DISPLAY_NAMES = {
    'Boro': 'Boro',
    'CaF2': 'CaF₂',
    'Saph': 'Sapph',
}

# Line dash styles for colorblind accessibility
MATERIAL_LINE_DASH = {
    'Boro': 'solid',
    'CaF2': 'dash',
    'Saph': 'dot',
}

SPECTRUM_LABEL_MAP = {
    'sky_aeri': 'Sky (AERI)',
    'sky_aeri_bb': 'Sky (AERI+BB)',
    'solar_direct': 'Solar (ASTM G-173 Direct)',
    'solar_global_tilt': 'Solar (ASTM G-173 Global)',
}


def blackbody_spectrum(wavelength_nm, temperature_k):
    """
    Calculate blackbody spectral radiance using Planck's law
    
    Args:
        wavelength_nm: wavelength in nanometers
        temperature_k: temperature in Kelvin
    
    Returns:
        spectral radiance in W/m²/sr/m
    """
    # Constants
    h = 6.626e-34  # Planck constant (J·s)
    c = 3e8        # Speed of light (m/s)
    k = 1.381e-23  # Boltzmann constant (J/K)
    sigma = 5.67e-8  # Stefan-Boltzmann constant (W/m²/K⁴)
    
    # Convert wavelength to meters
    wavelength_m = wavelength_nm * 1e-9
    
    # Planck's law
    numerator = 2 * h * c**2
    denominator = wavelength_m**5 * (np.exp(h * c / (wavelength_m * k * temperature_k)) - 1)
    
    return numerator / denominator


# Cache for loaded spectrum data
_spectrum_cache = {}


def load_sky_aeri_data():
    """
    Load sky spectrum AERI data from CSV file.
    Returns dataframe with wl_um, radiance, and data_source columns.
    """
    if 'sky_aeri' not in _spectrum_cache:
        csv_path = Path(__file__).parent / 'data' / 'sky_spectrum_aeri_20250703.011547.csv'
        df = pd.read_csv(csv_path)
        _spectrum_cache['sky_aeri'] = df
    return _spectrum_cache['sky_aeri']


def load_solar_data():
    """
    Load solar spectrum data from ASTM G173 CSV file.
    Returns dataframe with wavelength and irradiance columns.
    """
    if 'solar' not in _spectrum_cache:
        csv_path = Path(__file__).parent / 'data' / 'solar_astmg173.csv'
        
        # Read the CSV, skipping the header rows
        df = pd.read_csv(csv_path, skiprows=1, delimiter=';')
        
        # Clean column names
        df.columns = ['wavelength_nm', 'etr', 'global_tilt', 'direct_circumsolar', 'extra']
        
        # Remove any extra columns
        df = df[['wavelength_nm', 'global_tilt', 'direct_circumsolar']]
        
        # Convert string values to float, handling the comma decimal separator
        for col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
        
        _spectrum_cache['solar'] = df
    return _spectrum_cache['solar']


def create_unified_wavelength_grid():
    """
    Create a unified wavelength grid that includes:
    - Exact wavelength points from sky AERI data
    - Exact wavelength points from solar data
    - Blackbody linspace points (0.1 to 100 µm)
    
    Returns:
        np.array: Sorted array of unique wavelength values in micrometers
    """
    wavelengths = []
    
    # Add blackbody linspace points
    wavelengths.extend(np.linspace(0.1, 100, 1000))
    
    # Add exact sky AERI wavelengths
    try:
        sky_df = load_sky_aeri_data()
        wavelengths.extend(sky_df['wl_um'].values)
    except Exception:
        pass  # If sky data not available, continue
    
    # Add exact solar wavelengths
    try:
        solar_df = load_solar_data()
        # Convert nm to µm
        solar_wavelengths_um = solar_df['wavelength_nm'].values / 1000
        wavelengths.extend(solar_wavelengths_um)
    except Exception:
        pass  # If solar data not available, continue
    
    # Convert to numpy array, sort, and remove duplicates
    wavelengths = np.array(wavelengths)
    wavelengths = np.unique(wavelengths)
    wavelengths = wavelengths[wavelengths > 0]  # Remove any zero or negative values
    
    return wavelengths


def calculate_spectrum(spectrum_identifier):
    """
    Calculate spectrum radiance based on identifier.
    
    Args:
        spectrum_identifier: Can be:
            - float/int: Temperature in Celsius for blackbody
            - 'sky_aeri': Sky spectrum using only AERI data points
            - 'sky_aeri_bb': Sky spectrum using both AERI and BB data points
            - 'solar_global_tilt': Solar global tilt spectrum
            - 'solar_direct': Solar direct + circumsolar spectrum
    
    Returns:
        tuple: (wavelengths_um, radiance) where radiance is in W/m²/sr/m
    """
    
    # Get the unified wavelength grid
    wavelengths_um = create_unified_wavelength_grid()
    
    # Check if it's a numeric value (blackbody temperature)
    try:
        temp_c = float(spectrum_identifier)
        # Blackbody spectrum
        wavelengths_nm = wavelengths_um * 1000
        temp_k = temp_c + 273.15
        radiance = blackbody_spectrum(wavelengths_nm, temp_k)
        return wavelengths_um, radiance
    except (ValueError, TypeError):
        pass
    
    # Handle string identifiers
    if spectrum_identifier == 'sky_aeri':
        # Load sky AERI data and use only 'aeri' labeled points
        df = load_sky_aeri_data()
        df_aeri = df[df['data_source'] == 'aeri'].copy()
        df_aeri = df_aeri.sort_values('wl_um')
        
        # Interpolate to unified wavelength grid
        interp_func = interp1d(df_aeri['wl_um'], df_aeri['radiance'], 
                              kind='linear', bounds_error=False, fill_value=0)
        radiance_per_um = interp_func(wavelengths_um)
        
        # Convert from W/m²/sr/μm to W/m²/sr/m
        radiance = radiance_per_um * 1e6
        return wavelengths_um, radiance
        
    elif spectrum_identifier == 'sky_aeri_bb':
        # Load sky AERI data and use all points (both 'aeri' and 'bb')
        df = load_sky_aeri_data()
        df = df.sort_values('wl_um')
        
        # Interpolate to unified wavelength grid
        interp_func = interp1d(df['wl_um'], df['radiance'], 
                              kind='linear', bounds_error=False, fill_value=0)
        radiance_per_um = interp_func(wavelengths_um)
        
        # Convert from W/m²/sr/μm to W/m²/sr/m
        radiance = radiance_per_um * 1e6
        return wavelengths_um, radiance
        
    elif spectrum_identifier in ['solar_global_tilt', 'solar_direct']:
        # Load solar data
        df = load_solar_data()
        
        # Select appropriate column
        if spectrum_identifier == 'solar_global_tilt':
            irradiance_col = 'global_tilt'
        else:  # solar_direct
            irradiance_col = 'direct_circumsolar'
        
        # Convert wavelength from nm to μm
        wavelengths_um_data = df['wavelength_nm'] / 1000
        
        # Get irradiance in W/m²/nm
        irradiance_per_nm = df[irradiance_col].values
        
        # Convert from W/m²/nm to W/m²/μm (multiply by 1000)
        irradiance_per_um = irradiance_per_nm * 1000
        
        # Convert to radiance: divide by 2π steradians for hemisphere
        # (solar irradiance is typically given as total over hemisphere)
        radiance_per_um = irradiance_per_um / (2 * np.pi)
        
        # Interpolate to unified wavelength grid
        interp_func = interp1d(wavelengths_um_data, radiance_per_um, 
                              kind='linear', bounds_error=False, fill_value=0)
        radiance_per_um_interp = interp_func(wavelengths_um)
        
        # Convert from W/m²/sr/μm to W/m²/sr/m
        radiance = radiance_per_um_interp * 1e6
        return wavelengths_um, radiance
        
    else:
        raise ValueError(f"Unknown spectrum identifier: {spectrum_identifier}")




def calculate_spectrum_reflection(material_df, wavelengths_um, spectrum_radiance):
    """
    Calculate the percentage of spectrum radiation reflected by a material
    
    Args:
        material_df: DataFrame with 'wl' (wavelength in µm) and 'r' (reflectivity 0-1)
        wavelengths_um: array of wavelengths in micrometers
        spectrum_radiance: array of spectral radiance values
    
    Returns:
        Reflection percentage (0-100)
    """
    
    # Interpolate reflectivity data to match wavelength grid
    # Assume 0 reflectivity outside measured range
    interp_func = interp1d(material_df['wl'], material_df['r'], 
                          kind='linear', bounds_error=False, fill_value=0)
    reflectivity = interp_func(wavelengths_um)
    
    # Convert wavelengths to nm for integration
    wavelengths_nm = wavelengths_um * 1000
    
    # Calculate reflected power: integral of (spectrum * reflectivity) over wavelength
    reflected_power = np.trapz(spectrum_radiance * reflectivity, wavelengths_nm)
    
    # Calculate total spectrum power: integral of spectrum over wavelength
    total_spectrum_power = np.trapz(spectrum_radiance, wavelengths_nm)
    
    # Return percentage
    return (reflected_power / total_spectrum_power) * 100

def calculate_spectrum_transmission(material_df, wavelengths_um, spectrum_radiance):
    """
    Calculate the percentage of spectrum radiation transmitted by a material
    
    Args:
        material_df: DataFrame with 'wl' (wavelength in µm) and 't' (transmission 0-1)
        wavelengths_um: array of wavelengths in micrometers
        spectrum_radiance: array of spectral radiance values
    
    Returns:
        Transmission percentage (0-100)
    """
    
    # Interpolate transmission data to match wavelength grid
    # Assume 0 transmission outside measured range
    interp_func = interp1d(material_df['wl'], material_df['t'], 
                          kind='linear', bounds_error=False, fill_value=0)
    transmission = interp_func(wavelengths_um)
    
    # Convert wavelengths to nm for integration
    wavelengths_nm = wavelengths_um * 1000
    
    # Calculate transmitted power: integral of (spectrum * transmission) over wavelength
    transmitted_power = np.trapz(spectrum_radiance * transmission, wavelengths_nm)
    
    # Calculate total spectrum power: integral of spectrum over wavelength
    total_spectrum_power = np.trapz(spectrum_radiance, wavelengths_nm)
    
    # Return percentage
    return (transmitted_power / total_spectrum_power) * 100

def load_transmission_data(data_dir="data/materials"):
    """Load transmission CSV files for each material"""
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Error: Data directory '{data_dir}' not found.")
        return {}
    
    transmission_data = {}
    
    # Look for transmission files
    for csv_file in data_path.glob("*_transmission.csv"):
        # Extract material name from filename
        filename = csv_file.stem
        material = filename.split("_transmission")[0]
        
        try:
            df = pd.read_csv(csv_file)
            # Convert column names to standard format
            df.columns = ['wl', 't']  # wavelength, transmission
            # Convert transmission from percentage to fraction (0-1)
            df['t'] = df['t'] / 100.0
            
            transmission_data[material] = df
            print(f"Loaded transmission data for {material}: {len(df)} data points")
            
        except Exception as e:
            print(f"Error reading {csv_file.name}: {e}")
            continue
    
    return transmission_data

def load_and_combine_data(data_dir="data/processed"):
    """Load all CSV files from data directory and combine by material"""
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Error: Data directory '{data_dir}' not found.")
        return {}
    
    material_data = {}
    
    # Process all CSV files
    for csv_file in data_path.glob("*.csv"):
        # Extract material name from filename
        filename = csv_file.stem
        if "_reflectivity_" in filename:
            material = filename.split("_reflectivity_")[0]
        else:
            print(f"Warning: Skipping file {csv_file.name} - unexpected format")
            continue
        
        # Load data
        try:
            df = pd.read_csv(csv_file)
            if 'wl' not in df.columns or 'r' not in df.columns:
                print(f"Warning: Skipping file {csv_file.name} - missing required columns")
                continue
            
            # Add to material data
            if material not in material_data:
                material_data[material] = []
            material_data[material].append(df)
            print(f"Loaded {len(df)} data points from {csv_file.name}")
            
        except Exception as e:
            print(f"Error reading {csv_file.name}: {e}")
            continue
    
    # Combine reflectivity data for each material
    combined_data = {}
    for material, df_list in material_data.items():
        if len(df_list) == 1:
            combined_data[material] = {'reflectivity': df_list[0]}
        else:
            # Combine multiple files for same material
            combined_df = pd.concat(df_list, ignore_index=True)
            # Sort by wavelength
            combined_df = combined_df.sort_values('wl')
            # Remove duplicates if any
            combined_df = combined_df.drop_duplicates(subset=['wl'])
            combined_data[material] = {'reflectivity': combined_df}
            print(f"Combined {len(df_list)} files for {material}: {len(combined_df)} total points")
    
    # Load transmission data from parent directory
    parent_dir = str(data_path.parent)
    transmission_data = load_transmission_data(parent_dir)
    
    # Add transmission data to existing materials
    for material, trans_df in transmission_data.items():
        if material in combined_data:
            combined_data[material]['transmission'] = trans_df
        else:
            # Material only has transmission data
            combined_data[material] = {'transmission': trans_df}
    
    return combined_data

def create_reflectivity_plot(material_data, spectrum_identifier=None):
    """Create plotly figure with reflectivity data and optional spectrum curve"""
    fig = go.Figure()
    
    # Calculate spectrum and percentages if identifier provided
    percentages = {}
    wavelengths_um = None
    spectrum_radiance = None
    
    if spectrum_identifier is not None:
        # Calculate spectrum using unified function
        wavelengths_um, spectrum_radiance = calculate_spectrum(spectrum_identifier)
        
        # Calculate percentages for each material
        for material, data in material_data.items():
            percentages[material] = {}
            
            # Calculate reflection percentage
            percentages[material]['reflection'] = calculate_spectrum_reflection(data['reflectivity'], wavelengths_um, spectrum_radiance)
            percentages[material]['transmission'] = calculate_spectrum_transmission(data['transmission'], wavelengths_um, spectrum_radiance)
            
            # Calculate absorption and emissivity
            reflection = percentages[material].get('reflection', 0)
            transmission = percentages[material].get('transmission', 0)
            absorption = 100 - reflection - transmission
            emissivity = absorption / 100
            
            percentages[material]['absorption'] = absorption
            percentages[material]['emissivity'] = emissivity
    
    # Add trace for each material
    for material, data in material_data.items():
        # Only plot reflectivity data if available
        df = data['reflectivity']
        color = MATERIAL_COLORS.get(material, '#808080')  # Default gray if material not in predefined colors
        
        # Create name with percentages if available
        if material in percentages:
            parts = [f'{material}']
            parts.append(f'R={percentages[material]["reflection"]:.1f}%')
            parts.append(f'T={percentages[material]["transmission"]:.1f}%')
            parts.append(f'A={percentages[material]["absorption"]:.1f}%')
            parts.append(f'ε={percentages[material]["emissivity"]:.2f}')
            # Include spectrum identifier in the name
            try:
                temp_val = float(spectrum_identifier)
                spectrum_label = f'BB {temp_val:.0f}°C'
            except (ValueError, TypeError):
                spectrum_label = SPECTRUM_LABEL_MAP.get(spectrum_identifier, str(spectrum_identifier))
            name = f'{parts[0]} ({", ".join(parts[1:])} @ {spectrum_label})'
        else:
            name = material
        
        fig.add_trace(go.Scatter(
            x=df['wl'],
            y=df['r'] * 100,  # Convert to percentage
            mode='lines',
            name=name,
            line=dict(color=color, width=6),
            hovertemplate=f'{material}<br>Wavelength: %{{x:.2f}} µm<br>Reflectivity: %{{y:.1f}}%<extra></extra>'
        ))
    
    # Add spectrum curve if requested
    if spectrum_identifier is not None and spectrum_radiance is not None:
        # Choose color and styling based on spectrum type
        try:
            float(spectrum_identifier)  # Check if it's a blackbody temperature
            color = 'darkred'
            dash = 'dash'
            curve_name = f'Blackbody {spectrum_identifier}°C'
        except (ValueError, TypeError):
            # Handle other spectrum types
            if 'sky' in str(spectrum_identifier).lower():
                color = 'darkblue'
                dash = 'dot'
            elif 'solar' in str(spectrum_identifier).lower():
                color = 'orange'
                dash = 'dashdot'
            else:
                color = 'gray'
                dash = 'solid'
            
            curve_name = SPECTRUM_LABEL_MAP.get(spectrum_identifier, str(spectrum_identifier))
        
        # Add spectrum trace on secondary y-axis
        fig.add_trace(go.Scatter(
            x=wavelengths_um,
            y=spectrum_radiance,
            mode='lines',
            name=curve_name,
            line=dict(color=color, width=6, dash=dash),
            yaxis='y2',
            hovertemplate=f'{curve_name}<br>Wavelength: %{{x:.2f}} µm<br>Radiance: %{{y:.2e}} W/m²/sr/m<extra></extra>'
        ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Material Reflectivity vs Wavelength',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 28}
        },
        xaxis_title="Wavelength (µm)",
        yaxis_title="Reflectivity (%)",
        xaxis=dict(
            type='linear',  # Linear scale for wavelength
            tickfont=dict(size=24),
            title_font=dict(size=24),
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
            range=[0, 20],  # 0 to 20 µm
            constraintoward='left',
            rangemode='tozero'
        ),
        yaxis=dict(
            tickfont=dict(size=24),
            title_font=dict(size=24),
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
            range=[0, 100],  # 0 to 100%
            constraintoward='bottom',
            rangemode='tozero'
        ),
        font=dict(family="Times New Roman, Times, serif", color="black"),
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=1200,
        height=600,
        hovermode='x unified',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.02,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
            font=dict(size=24)
        ),
        margin=dict(l=80, r=80, t=100, b=80),
        dragmode='zoom'
    )
    
    # Add secondary y-axis if spectrum curve is present
    if spectrum_identifier is not None and spectrum_radiance is not None:
        # Find max value in the visible range (0-20 µm) to scale secondary axis properly
        visible_mask = wavelengths_um <= 20
        if len(spectrum_radiance[visible_mask]) > 0:
            max_radiance_value = np.max(spectrum_radiance[visible_mask])
        else:
            max_radiance_value = np.max(spectrum_radiance) if len(spectrum_radiance) > 0 else 1
        
        # Determine axis title based on spectrum type
        try:
            float(spectrum_identifier)
            y2_title = "Blackbody Spectral Radiance (W/m²/sr/m)"
        except (ValueError, TypeError):
            if 'sky' in str(spectrum_identifier).lower():
                y2_title = "Sky Spectral Radiance (W/m²/sr/m)"
            elif 'solar' in str(spectrum_identifier).lower():
                y2_title = "Solar Spectral Radiance (W/m²/sr/m)"
            else:
                y2_title = "Spectral Radiance (W/m²/sr/m)"
        
        fig.update_layout(
            yaxis2=dict(
                title=dict(
                    text=y2_title,
                    font=dict(size=24)
                ),
                tickfont=dict(size=24),
                overlaying='y',
                side='right',
                range=[0, max_radiance_value * 1.1],  # Add 10% margin
                showgrid=False
            )
        )
    
    # Add minor gridlines
    fig.update_xaxes(minor=dict(showgrid=True, gridwidth=0.5, gridcolor='rgba(128,128,128,0.1)'))
    
    return fig

def gentable(temperatures=None, data_dir="data/processed"):
    """
    Generate LaTeX table with material properties at given spectra.
    
    Args:
        temperatures: List of spectrum identifiers. Can be:
            - float/int: Temperature in Celsius for blackbody
            - 'sky_aeri': Sky spectrum using only AERI data points
            - 'sky_aeri_bb': Sky spectrum using both AERI and BB data points
            - 'solar_global_tilt': Solar global tilt spectrum
            - 'solar_direct': Solar direct + circumsolar spectrum
            Default: ['sky_aeri_bb', 15, 30, 45, 60, 'solar_direct', 'solar_global_tilt']
        data_dir: Directory containing processed CSV files (default: "data/processed")
    """
    # Use default if not specified
    if temperatures is None:
        temperatures = ['sky_aeri_bb', 15, 30, 45, 60, 'solar_global_tilt']
    # Load material data
    material_data = load_and_combine_data(data_dir)
    
    if not material_data:
        print("No data found.")
        return
    
    # Material name mapping for LaTeX
    material_names = {
        'Boro': 'Borosilicate',
        'CaF2': 'CaF$_2$',
        'Saph': 'Sapphire'
    }
    
    # Material name mapping for plain text table
    material_names_plain = {
        'Boro': 'Borosilicate',
        'CaF2': 'CaF2',
        'Saph': 'Sapphire'
    }
    
    # Start LaTeX table
    latex_output = []
    latex_output.append("\\begin{tabular}{|l|c|c|c|c|c|}")
    latex_output.append("\\hline")
    latex_output.append("\\textbf{Material} & \\textbf{Spectrum} & \\textbf{Transmission} & \\textbf{Reflectance} & \\textbf{Absorption} & \\textbf{Emissivity ($\\varepsilon$)} \\\\")
    latex_output.append("\\hline")
    
    # Data for tabulate table
    table_data = []
    
    # Process materials in specific order: Boro, Saph, CaF2
    material_order = ['Boro', 'Saph', 'CaF2']
    
    # Process each material in order
    for material_code in material_order:
        if material_code not in material_data or material_code not in material_names:
            continue
            
        data = material_data[material_code]
        material_name = material_names[material_code]
        material_name_plain = material_names_plain[material_code]
        
        # Process each spectrum for this material
        for i, spectrum_id in enumerate(temperatures):
            # Calculate spectrum
            wavelengths_um, spectrum_radiance = calculate_spectrum(spectrum_id)
            
            # Calculate properties
            reflection = calculate_spectrum_reflection(data['reflectivity'], wavelengths_um, spectrum_radiance)
            transmission = calculate_spectrum_transmission(data['transmission'], wavelengths_um, spectrum_radiance)
            
            absorption = 100 - reflection - transmission
            emissivity = absorption / 100
            
            # Format spectrum label for LaTeX
            try:
                temp_val = float(spectrum_id)
                spectrum_label = f"{temp_val:.0f} $^\\circ$C"
            except (ValueError, TypeError):
                # Format string identifiers nicely
                spectrum_label = SPECTRUM_LABEL_MAP.get(spectrum_id, str(spectrum_id))
            
            # Format the LaTeX row
            if i == 0:
                # First row for this material - include material name
                row = f"{material_name} & {spectrum_label} & {transmission:.1f}\\% & {reflection:.1f}\\% & {absorption:.1f}\\% & {emissivity:.2f} \\\\"
            else:
                # Subsequent rows - empty material name column
                row = f"                              & {spectrum_label} & {transmission:.1f}\\% & {reflection:.1f}\\% & {absorption:.1f}\\% & {emissivity:.2f} \\\\"
            
            latex_output.append(row)
            
            # Format spectrum label for plain text
            try:
                temp_val = float(spectrum_id)
                spectrum_label_plain = f"{temp_val:.0f}°C"
            except (ValueError, TypeError):
                # Format string identifiers nicely
                spectrum_label_plain = SPECTRUM_LABEL_MAP.get(spectrum_id, str(spectrum_id))
            
            # Add data for tabulate table
            table_data.append([
                material_name_plain if i == 0 else "",
                spectrum_label_plain,
                f"{transmission:.1f}%",
                f"{reflection:.1f}%",
                f"{absorption:.1f}%",
                f"{emissivity:.2f}"
            ])
        
        # Add hline after each material
        latex_output.append("\\hline")
    
    # End table
    latex_output.append("\\end{tabular}")
    
    # Print LaTeX table
    print("\nLaTeX Table:")
    print("="*60)
    for line in latex_output:
        print(line)
    
    # Print tabulate table
    print("\n\nFormatted Table:")
    print("="*60)
    headers = ["Material", "Spectrum", "Transmission", "Reflectance", "Absorption", "Emissivity (ε)"]
    print(tabulate(table_data, headers=headers, tablefmt="simple"))


def load_refrindex_data(data_dir="data/materials"):
    """Load refractive index CSV files for each material"""
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Error: Data directory '{data_dir}' not found.")
        return {}
    
    refrindex_data = {}
    
    # Look for refractive index files
    for csv_file in data_path.glob("*_refrindex_*.csv"):
        # Extract material name from filename
        filename = csv_file.stem
        material = filename.split("_refrindex_")[0]
        
        try:
            df = pd.read_csv(csv_file)
            # Ensure columns are properly named
            if 'wl' in df.columns and 'n' in df.columns:
                # If material already exists, combine data
                if material in refrindex_data:
                    # Append and sort by wavelength
                    combined_df = pd.concat([refrindex_data[material], df], ignore_index=True)
                    combined_df = combined_df.sort_values('wl').drop_duplicates(subset=['wl'])
                    refrindex_data[material] = combined_df
                else:
                    refrindex_data[material] = df
                print(f"Loaded refractive index data for {material}: {len(df)} data points from {csv_file.name}")
            else:
                print(f"Warning: {csv_file.name} missing required columns 'wl' and 'n'")
                
        except Exception as e:
            print(f"Error reading {csv_file.name}: {e}")
            continue
    
    return refrindex_data

def refrindex(data_dir="data/materials"):
    """
    Plot refractive index vs wavelength for all materials.
    
    Args:
        data_dir: Directory containing refractive index CSV files (default: "data")
    """
    # Load refractive index data
    refrindex_data = load_refrindex_data(data_dir)
    
    if not refrindex_data:
        print("No refractive index data found.")
        return
    
    # Create plotly figure
    fig = go.Figure()
    
    # Add trace for each material
    for material, df in refrindex_data.items():
        color = MATERIAL_COLORS.get(material, '#808080')  # Default gray if material not in predefined colors
        
        fig.add_trace(go.Scatter(
            x=df['wl'],
            y=df['n'],
            mode='lines',
            name=material,
            line=dict(color=color, width=3),
            hovertemplate=f'{material}<br>Wavelength: %{{x:.3f}} µm<br>Refractive Index: %{{y:.4f}}<extra></extra>'
        ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Material Refractive Index vs Wavelength',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 28}
        },
        xaxis_title="Wavelength (µm)",
        yaxis_title="Refractive Index (n)",
        xaxis=dict(
            type='log',  # Log scale for wavelength
            tickfont=dict(size=16),
            title_font=dict(size=20),
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
            range=[-1, 3],  # 0.1 to 1000 µm
        ),
        yaxis=dict(
            tickfont=dict(size=16),
            title_font=dict(size=20),
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
            rangemode='tozero'
        ),
        font=dict(family="Times New Roman, Times, serif", color="black"),
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=1200,
        height=800,
        hovermode='x unified',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.02,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
            font=dict(size=16)
        ),
        margin=dict(l=80, r=80, t=100, b=80),
        dragmode='zoom'
    )
    
    # Add minor gridlines
    fig.update_xaxes(minor=dict(showgrid=True, gridwidth=0.5, gridcolor='rgba(128,128,128,0.1)'))
    fig.update_yaxes(minor=dict(showgrid=True, gridwidth=0.5, gridcolor='rgba(128,128,128,0.1)'))
    
    # Save to temporary HTML file and open
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
        html_content = fig.to_html(
            include_plotlyjs='cdn',
            config={
                'displayModeBar': True,
                'scrollZoom': True,
                'doubleClick': 'reset+autosize',
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': 'refrindex_plot',
                    'height': 800,
                    'width': 1200,
                    'scale': 2
                }
            }
        )
        
        # Add some styling
        styled_html = html_content.replace(
            '<head>',
            '''<head>
            <style>
                body {
                    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                    margin: 0;
                    padding: 20px;
                    font-family: Arial, sans-serif;
                }
                .plotly-graph-div {
                    border-radius: 10px;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
                    background: white;
                    padding: 10px;
                }
            </style>'''
        )
        
        f.write(styled_html)
        temp_file = f.name
    
    # Open in browser
    webbrowser.open(f'file://{temp_file}')
    print(f"Refractive index plot saved and opened: {temp_file}")

def genlist(data_dir="data/processed"):
    """
    Generate transmission, reflection, and absorption lists for each material
    at temperatures from 0°C to 100°C for both blackbody and sky radiance.
    """
    # Load material data
    material_data = load_and_combine_data(data_dir)

    if not material_data:
        print("No data found.")
        return

    # Create temperature range from 0°C to 100°C
    temperatures = list(range(101))  # 0 to 100 inclusive

    # Create wavelength grid
    wavelengths_um = np.linspace(0.1, 100, 1000)
    wavelengths_nm = wavelengths_um * 1000

    # Initialize dictionaries for results
    bb_results = {}
    sky_results = {}

    # Process each material
    for material, data in material_data.items():
        print(f"Processing {material}...")

        # Initialize lists for this material
        bb_results[material] = {'refl': [], 'trans': [], 'abs': []}
        sky_results[material] = {'refl': [], 'trans': [], 'abs': []}

        # Calculate for each temperature
        for temp_c in temperatures:
            temp_k = temp_c + 273.15

            # Blackbody calculations
            bb_radiance = blackbody_spectrum(wavelengths_nm, temp_k)

            # Sky radiance calculations
            sky_radiance = calculate_sky_radiance(wavelengths_um, temp_c)

            # Calculate reflection and transmission for blackbody
            bb_reflection = calculate_spectrum_reflection(data['reflectivity'], wavelengths_um, bb_radiance) / 100
            bb_transmission = calculate_spectrum_transmission(data['transmission'], wavelengths_um, bb_radiance) / 100
            bb_absorption = 1.0 - bb_reflection - bb_transmission

            # Calculate reflection and transmission for sky
            sky_reflection = calculate_spectrum_reflection(data['reflectivity'], wavelengths_um, sky_radiance) / 100
            sky_transmission = calculate_spectrum_transmission(data['transmission'], wavelengths_um, sky_radiance) / 100
            sky_absorption = 1.0 - sky_reflection - sky_transmission

            # Store results (rounded to 4 decimals)
            bb_results[material]['refl'].append(round(bb_reflection, 4))
            bb_results[material]['trans'].append(round(bb_transmission, 4))
            bb_results[material]['abs'].append(round(bb_absorption, 4))

            sky_results[material]['refl'].append(round(sky_reflection, 4))
            sky_results[material]['trans'].append(round(sky_transmission, 4))
            sky_results[material]['abs'].append(round(sky_absorption, 4))

    # Print results in Python-pasteable format
    print("\n# Blackbody emission results (0°C to 100°C) - values as fractions 0-1")
    print("bb_data = {")
    for material, values in bb_results.items():
        print(f"    '{material}': {{")
        print(f"        'refl': {values['refl']},")
        print(f"        'trans': {values['trans']},")
        print(f"        'abs': {values['abs']}")
        print(f"    }},")
    print("}")

    print("\n# Sky radiance results (0°C to 100°C ground air temperature) - values as fractions 0-1")
    print("sky_data = {")
    for material, values in sky_results.items():
        print(f"    '{material}': {{")
        print(f"        'refl': {values['refl']},")
        print(f"        'trans': {values['trans']},")
        print(f"        'abs': {values['abs']}")
        print(f"    }},")
    print("}")

def transmission(data_dir="data/materials", show_solar=True):
    """
    Plot transmission vs wavelength for all materials.

    Args:
        data_dir: Directory containing transmission CSV files (default: "data")
        show_solar: If True, overlay solar spectrum on secondary y-axis
    """
    # Load transmission data
    transmission_data = load_transmission_data(data_dir)
    
    if not transmission_data:
        print("No transmission data found.")
        return
    
    # Create figure with subplots if showing solar
    if show_solar:
        from plotly.subplots import make_subplots
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.12,
            row_heights=[0.5, 0.5],
        )
    else:
        fig = go.Figure()

    # Add trace for each material
    for material, df in transmission_data.items():
        color = MATERIAL_COLORS.get(material, '#808080')  # Default gray if material not in predefined colors
        display_name = MATERIAL_DISPLAY_NAMES.get(material, material)
        dash_style = MATERIAL_LINE_DASH.get(material, 'solid')

        # Extend lines at 0% transmission to 50 µm
        df_plot = df.copy()
        if df_plot['t'].iloc[-1] < 0.01:  # If last point is ~0%
            extension = pd.DataFrame({'wl': [50.0], 't': [0.0]})
            df_plot = pd.concat([df_plot, extension], ignore_index=True)

        fig.add_trace(go.Scatter(
            x=df_plot['wl'],
            y=df_plot['t'] * 100,  # Convert to percentage
            mode='lines',
            name=display_name,
            line=dict(color=color, width=6, dash=dash_style),
            hovertemplate=f'{display_name}<br>Wavelength: %{{x:.2f}} µm<br>Transmission: %{{y:.1f}}%<extra></extra>'
        ), row=1 if show_solar else None, col=1 if show_solar else None)

    # Add solar spectrum if requested
    if show_solar:
        from pathlib import Path
        solar_file = Path(__file__).parent / 'data' / 'solar_astmg173.csv'
        # Read with European decimal format (comma as decimal, semicolon as separator)
        solar_df = pd.read_csv(solar_file, sep=';', decimal=',', skiprows=2, header=None)
        # Columns: 0=wavelength(nm), 1=ETR, 2=Global tilt, 3=Direct
        solar_df['wl'] = solar_df[0] / 1000.0  # Convert nm to µm
        solar_df['irradiance'] = solar_df[2]  # Global tilt irradiance (W/m²/nm)

        # Plot λ · f(λ) for proper area representation on log scale
        solar_y = solar_df['wl'] * solar_df['irradiance']
        fig.add_trace(go.Scatter(
            x=solar_df['wl'],
            y=solar_y,
            mode='lines',
            name='Solar',
            line=dict(color='#B8860B', width=4),  # Dark yellow/goldenrod
            hovertemplate='Solar<br>Wavelength: %{x:.2f} µm<br>λ·Irradiance: %{y:.2f} W/m²<extra></extra>',
            showlegend=True,
            legend='legend2',
        ), row=2, col=1)

        # Add 60°C blackbody spectrum
        h = 6.626e-34  # Planck constant (J·s)
        c = 3e8        # Speed of light (m/s)
        k = 1.381e-23  # Boltzmann constant (J/K)
        T = 60 + 273.15  # 60°C in Kelvin

        # Generate wavelengths from 0.28 to 50 µm
        bb_wl_um = np.linspace(0.28, 50, 1000)
        bb_wl_m = bb_wl_um * 1e-6  # Convert to meters

        # Planck function: B(λ,T) = (2hc²/λ⁵) / (exp(hc/λkT) - 1)
        # Units: W/m²/sr/m
        exponent = (h * c) / (bb_wl_m * k * T)
        bb_radiance = (2 * h * c**2 / bb_wl_m**5) / (np.exp(exponent) - 1)
        # Convert to W/m²/sr/nm (divide by 1e9)
        bb_radiance_nm = bb_radiance / 1e9

        # Convert radiance to exitance (W/m²/nm) by multiplying by π
        bb_exitance = bb_radiance_nm * np.pi
        # Plot λ · f(λ) for proper area representation on log scale
        bb_scaled = bb_wl_um * bb_exitance

        # Calculate combined max for both y-axes
        solar_max = solar_y.max()
        bb_max = bb_scaled.max()
        y_max = max(solar_max, bb_max) * 1.05  # 5% margin

        fig.add_trace(go.Scatter(
            x=bb_wl_um,
            y=bb_scaled,
            mode='lines',
            name='60°C BB',
            line=dict(color='#8B0000', width=6, dash='dash'),  # Dark red, thicker
            hovertemplate='60°C Blackbody<br>Wavelength: %{x:.2f} µm<br>λ·Exitance: %{y:.2f} W/m²<extra></extra>',
            showlegend=True,
            legend='legend2',
        ), row=2, col=1)

    # Update layout (no title - Nature style puts this in caption)
    fig_height = 900 if show_solar else 520

    fig.update_layout(
        font=dict(family="Times New Roman, Times, serif", color="black"),
        plot_bgcolor='#f5f5f5',
        paper_bgcolor='white',
        width=1200,
        height=fig_height,
        hovermode='x unified',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98 if not show_solar else 1.0,
            xanchor="right",
            x=0.98,
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=35),
            itemwidth=50
        ),
        legend2=dict(
            orientation="v",
            yanchor="top",
            y=0.44,
            xanchor="right",
            x=0.98,
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=35),
            itemwidth=50
        ),
        margin=dict(l=100, r=140 if show_solar else 80, t=40, b=130),
        dragmode='zoom'
    )

    # Configure axes
    xaxis_config = dict(
        type='log',
        tickfont=dict(size=39, family="Times New Roman, Times, serif"),
        title_font=dict(size=42, family="Times New Roman, Times, serif"),
        showgrid=True,
        gridcolor='rgba(200,200,200,0.5)',
        gridwidth=1,
        range=[np.log10(0.23), np.log10(50)],
        tickmode='array',
        tickvals=[0.3, 0.5, 1, 2, 3, 5, 10, 20, 50],
        ticktext=['0.3', '0.5', '1', '2', '3', '5', '10', '20', '50'],
    )

    yaxis_trans_config = dict(
        title="Transmission (%)",
        tickfont=dict(size=39, family="Times New Roman, Times, serif"),
        title_font=dict(size=42, family="Times New Roman, Times, serif"),
        showgrid=True,
        gridcolor='rgba(200,200,200,0.5)',
        gridwidth=1,
        dtick=20,
        range=[0, 100],
    )

    if show_solar:
        # Two-panel layout
        fig.update_xaxes(xaxis_config, row=1, col=1)
        fig.update_xaxes(
            **xaxis_config,
            title=dict(text="Wavelength (µm)", font=dict(size=42, family="Times New Roman, Times, serif")),
            row=2, col=1
        )
        fig.update_yaxes(yaxis_trans_config, row=1, col=1)
        fig.update_yaxes(
            title="λE(λ) (W/m²)",
            tickfont=dict(size=39, family="Times New Roman, Times, serif"),
            title_font=dict(size=42, family="Times New Roman, Times, serif"),
            showgrid=True,
            gridcolor='rgba(200,200,200,0.5)',
            gridwidth=1,
            range=[0, y_max],
            row=2, col=1
        )
    else:
        # Single panel
        fig.update_layout(
            xaxis={**xaxis_config, "title": dict(text="Wavelength (µm)", font=dict(size=42, family="Times New Roman, Times, serif"))},
            yaxis=yaxis_trans_config
        )

    # Save to figures directory
    import subprocess
    from pathlib import Path

    figures_dir = Path(__file__).parent / 'figures'
    figures_dir.mkdir(exist_ok=True)

    pdf_file = figures_dir / 'transmissions.pdf'
    fig.write_image(str(pdf_file), format='pdf', width=1200, height=fig_height)
    print(f"Transmission plot saved: {pdf_file}")


def plot_r_vs_temp(data_dir="data/processed"):
    """
    Plot reflection percentage vs blackbody temperature for all materials.
    
    Args:
        data_dir: Directory containing processed CSV files (default: "data/processed")
    """
    # Load material data
    material_data = load_and_combine_data(data_dir)
    
    if not material_data:
        print("No data found to plot.")
        return
    
    # Generate temperature range from -20°C to +100°C, every 0.5°C
    temperatures = np.arange(-270, 6000, 1)
    
    # Create figure
    fig = go.Figure()
    
    # Calculate reflection percentages for each material
    for material, data in material_data.items():
        df = data['reflectivity']
        reflections = []
        
        # Calculate reflection for each temperature
        for temp_c in temperatures:
            # Create wavelength grid and blackbody spectrum for this temperature
            wavelengths_um = np.linspace(0.1, 100, 1000)
            wavelengths_nm = wavelengths_um * 1000
            temp_k = temp_c + 273.15
            bb_radiance = blackbody_spectrum(wavelengths_nm, temp_k)
            reflection_pct = calculate_spectrum_reflection(df, wavelengths_um, bb_radiance)
            reflections.append(reflection_pct)
        
        # Get color for material
        color = MATERIAL_COLORS.get(material, '#808080')
        
        # Add trace for this material
        fig.add_trace(go.Scatter(
            x=temperatures,
            y=reflections,
            mode='lines',
            name=material,
            line=dict(color=color, width=3),
            hovertemplate=f'{material}<br>Temperature: %{{x:.1f}}°C<br>Reflection: %{{y:.2f}}%<extra></extra>'
        ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Material Reflection vs Blackbody Temperature',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 28}
        },
        xaxis_title="Blackbody Temperature (°C)",
        yaxis_title="Total Reflection of Blackbody Emission (%)",
        xaxis=dict(
            tickfont=dict(size=16),
            title_font=dict(size=20),
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)'
        ),
        yaxis=dict(
            tickfont=dict(size=16),
            title_font=dict(size=20),
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
            rangemode='tozero'
        ),
        font=dict(family="Times New Roman, Times, serif", color="black"),
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=1200,
        height=800,
        hovermode='x unified',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.02,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
            font=dict(size=16)
        ),
        margin=dict(l=80, r=80, t=100, b=80),
        dragmode='zoom'
    )
    
    # Add minor gridlines
    fig.update_xaxes(minor=dict(showgrid=True, gridwidth=0.5, gridcolor='rgba(128,128,128,0.1)'))
    fig.update_yaxes(minor=dict(showgrid=True, gridwidth=0.5, gridcolor='rgba(128,128,128,0.1)'))
    
    # Save to temporary HTML file and open
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
        html_content = fig.to_html(
            include_plotlyjs='cdn',
            config={
                'displayModeBar': True,
                'scrollZoom': True,
                'doubleClick': 'reset+autosize',
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': 'r_vs_temp_plot',
                    'height': 800,
                    'width': 1200,
                    'scale': 2
                }
            }
        )
        
        # Add some styling
        styled_html = html_content.replace(
            '<head>',
            '''<head>
            <style>
                body {
                    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                    margin: 0;
                    padding: 20px;
                    font-family: Arial, sans-serif;
                }
                .plotly-graph-div {
                    border-radius: 10px;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
                    background: white;
                    padding: 10px;
                }
            </style>'''
        )
        
        f.write(styled_html)
        temp_file = f.name
    
    # Open in browser
    webbrowser.open(f'file://{temp_file}')
    print(f"Plot saved and opened: {temp_file}")

def plot(*files, data_dir="data/processed", bb=None):
    """
    Plot reflectivity data from processed CSV files.
    
    Args:
        *files: Optional specific files to plot. If not provided, plots all files in data_dir.
        data_dir: Directory containing processed CSV files (default: "data/processed")
        bb: Spectrum identifier. Can be:
            - float/int: Temperature in Celsius for blackbody (e.g., 25)
            - 'sky_aeri': Sky spectrum using only AERI data points
            - 'sky_aeri_bb': Sky spectrum using both AERI and BB data points
            - 'solar_global_tilt': Solar global tilt spectrum
            - 'solar_direct': Solar direct + circumsolar spectrum
    """
    # Parse bb parameter
    spectrum_identifier = bb
    if files:
        # If specific files are provided, use them
        material_data = {}
        for file_path in files:
            path = Path(file_path)
            if not path.exists():
                print(f"Error: File '{file_path}' not found.")
                continue
            
            # Extract material from filename
            filename = path.stem
            if "_reflectivity_" in filename:
                material = filename.split("_reflectivity_")[0]
            else:
                material = filename  # Use full filename as material name
            
            try:
                df = pd.read_csv(path)
                if material not in material_data:
                    material_data[material] = {'reflectivity': df}
                else:
                    # Combine with existing data
                    combined_df = pd.concat([material_data[material]['reflectivity'], df], ignore_index=True)
                    material_data[material]['reflectivity'] = combined_df.sort_values('wl').drop_duplicates(subset=['wl'])
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    else:
        # Load all files from data directory
        material_data = load_and_combine_data(data_dir)
    
    if not material_data:
        print("No data found to plot.")
        return
    
    # Create plot
    fig = create_reflectivity_plot(material_data, spectrum_identifier=spectrum_identifier)
    
    # Save to temporary HTML file and open
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
        html_content = fig.to_html(
            include_plotlyjs='cdn',
            config={
                'displayModeBar': True,
                'scrollZoom': True,
                'doubleClick': 'reset+autosize',
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': 'reflectivity_plot',
                    'height': 800,
                    'width': 1200,
                    'scale': 2
                }
            }
        )
        
        # Add some styling
        styled_html = html_content.replace(
            '<head>',
            '''<head>
            <style>
                body {
                    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                    margin: 0;
                    padding: 20px;
                    font-family: Arial, sans-serif;
                }
                .plotly-graph-div {
                    border-radius: 10px;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
                    background: white;
                    padding: 10px;
                }
            </style>'''
        )
        
        f.write(styled_html)
        temp_file = f.name
    
    # Open in browser
    webbrowser.open(f'file://{temp_file}')
    print(f"Plot saved and opened: {temp_file}")

if __name__ == '__main__':
    fire.Fire({
        'plot': plot,
        'r_vs_temp': plot_r_vs_temp,
        'gentable': gentable,
        'genlist': genlist,
        'refrindex': refrindex,
        'transmission': transmission
    })
