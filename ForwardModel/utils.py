import os
import xarray as xr
import numpy as np
from constants import g0, k, g_e
from coord_transforms import R_e, e2
import re 


def get_location_info():
    """Return dictionaries with location coordinates and names."""
    return {
        'kir': (67.87, 21.03),  # Kiruna
        'lyb': (78.15, 16.04),  # Longyearbyen
        'sod': (67.4, 26.6),    # Sodankyla
        'caj': (-6.9, -38.5),    # Cajazeiras
        'msh': (42.6, -71.5),     # Millstone Hill
        'car': (-7.38, -36.52), # Cariri
        'NL': (-2.5, 0.0) # some random ahh place
    }, {
        'kir': 'Kiruna',
        'lyb': 'Longyearbyen',
        'sod': 'Sodankyla',
        'caj': 'Cajazeiras',
        'msh': 'Millstone Hill',
        'NL': 'NatalieLoc'
    }


def extract_date(file_name):
    """Extract a unique identifier from the input file name."""
    basename = os.path.basename(file_name)

    # Pattern 1: For WACCM-X format like '...YYYY-MM-DD...'
    waccmx_match = re.search(r'(\d{4}-\d{2}-\d{2})', basename)
    if waccmx_match:
        return waccmx_match.group(1)

    # Pattern 2: For TIEGCM format, use the full timestamp for uniqueness
    tiegcm_match = re.search(r'_(\d{14})_', basename)
    if tiegcm_match:
        date_str = tiegcm_match.group(1)
        # Reformat from YYYYMMDDHHMMSS to YYYY-MM-DD_HHMMSS
        return f"{date_str[0:4]}-{date_str[4:6]}-{date_str[6:8]}_{date_str[8:14]}"

    # If no known date pattern is found, raise an error
    raise ValueError(f"Invalid file name format; could not extract date from '{basename}'")


def generate_output_path(input_file, location_name, output_dir=None):
    """Generate output file path based on input file and location."""
    # The default output directory is now handled by argparse in main.py

    if output_dir is None:
        output_dir = f"/rds/projects/r/reidb-waccm-x/generated_files/therm_wind/"

    # Create a subdirectory for the specific location to keep outputs organized
    location_output_dir = os.path.join(output_dir, location_name)
    os.makedirs(location_output_dir, exist_ok=True)

    date = extract_date(input_file)
    output_file_name = f"{location_name}_{date}_therm_winds.nc"
    return os.path.join(location_output_dir, output_file_name)


def add_los_variables(ds_final, los_data, times, distance):


    """Add LOS data variables to the dataset, separated by direction.
    
    Args:
        ds_final (xr.Dataset): The xarray Dataset to which variables will be added.
        los_data (dict): Dictionary mapping direction strings (e.g., 'north') to data dictionaries
                        containing keys: 'v_los', 'T', 'emission_rates', etc.
        times (array-like): Time coordinates for the dataset.
        distance (array-like): Distance coordinates for the dataset.
    
    Returns:
        xr.Dataset: The updated dataset with LOS variables added.
    """


    for direction, data in los_data.items():
        freq_bins = np.arange(len(data['f_range']))
        
        ds_final[f'LOS_wind_{direction}'] = xr.DataArray(
            data['v_los'],
            dims=['time', 'distance'],
            coords={'time': times, 'distance': distance},
            attrs={'units': 'm/s', 'long_name': f'Line of Sight Wind ({direction})'}
        )
        ds_final[f'LOS_T_{direction}'] = xr.DataArray(
            data['T'],
            dims=['time', 'distance'],
            coords={'time': times, 'distance': distance},
            attrs={'units': 'K', 'long_name': f'Temperature ({direction})'}
        )

        ds_final[f'wind_doppler_{direction}'] = xr.DataArray(
            data['wind_doppler'],
            dims=['time'],
            coords={'time': times},
            attrs={'units': 'm/s', 'long_name': f'Integrated Doppler Wind ({direction})'}
        )
        ds_final[f'temp_doppler_{direction}'] = xr.DataArray(
            data['temp_doppler'],
            dims=['time'],
            coords={'time': times},
            attrs={'units': 'K', 'long_name': f'Integrated Doppler Temperature ({direction})'}
        )
        ds_final[f'spectrum_{direction}'] = xr.DataArray(
            data['spectrum'],
            dims=['time', 'frequency_bins'],
            coords={'time': times, 'frequency_bins': freq_bins},
            attrs={'long_name': f'Normalized Doppler Spectrum ({direction})'}
        )
        ds_final[f'frequency_range_{direction}'] = xr.DataArray(
            data['f_range'],
            dims=['frequency_bins'],
            coords={'frequency_bins': freq_bins},
            attrs={'units': 'Hz', 'long_name': 'Frequency range for spectrum'}
        )
    return ds_final


## add Z3g to dataset
def add_Z3g(ds, lat=None):
    """
    Add geometric height (Z3g) to the dataset based on geopotential height (Z3).
    
    Parameters:
    ds : xarray.Dataset
        Dataset containing 'Z3' variable (geopotential height).
    
    Returns:
    xarray.Dataset
        Dataset with 'Z3g' variable added.
    """

    def geopotential_to_geometric(z_gp, lat=None):
        """
        Convert geopotential height (Z3) to geometric height (Z3g).
        
        Parameters:
        z_gp : array-like, geopotential height in meters
        lat : array-like, latitude in degrees (optional, for latitude-dependent gravity)
        
        Returns:
        z_g : geometric height in meters
        """

        phi = np.deg2rad(lat)
        sin2 = np.sin(phi)**2
        
        # Somigliana formula (normal gravity at sea-level)
        g = g_e * (1 + k * sin2) / np.sqrt(1 - e2 * sin2)
        
        #  conversion formula
        z_g = (R_e * z_gp) / (R_e - z_gp * g / g0)
        
        return z_g



    if 'Z3' not in ds:
        raise ValueError("Dataset must contain 'Z3' variable for geopotential height.")
    
    z_gp = ds['Z3'].values  # Geopotential height in meters
    
    z_g = geopotential_to_geometric(z_gp, lat)
    
    ds['Z3g'] = xr.DataArray(
        z_g,
        coords=ds['Z3'].coords,
        dims=ds['Z3'].dims,
        name='Z3g',
        attrs={'units': 'm', 'long_name': 'Geometric Height'}
    )
    
    return ds