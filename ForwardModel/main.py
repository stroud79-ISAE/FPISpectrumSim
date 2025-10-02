import sys
import logging
import xarray as xr
import numpy as np
import argparse
from doppler import DopplerProcessor, LineOfSightCalculator
from interpolation import interpolate_vertical
from redline import calculate_ver
from utils import add_los_variables, generate_output_path, get_location_info, add_Z3g
from tiegcm_wrapper import tiegcm_to_waccmx_structure

# init logging to see issues
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def process_dataset(ds, center_lat, center_lon, grid_spacing_m, grid_spacing_km, los_elevation_deg, los_max_distance_km, alt_init):
    """
    Processes model data by calculating VER, performing vertical interpolation,
    and then directly interpolating atmospheric data onto the line-of-sight (LOS)
    paths for Doppler processing.

    Inputs:
    ====================================================================
    - ds (xarray.Dataset): Input dataset (WACCM-X native).
    - center_lat (float): Latitude of the center location.
    - center_lon (float): Longitude of the center location.
    - grid_spacing_m (float): Grid spacing in meters, used for vertical interpolation and LOS path steps.
    - grid_spacing_km (float): Grid spacing in kilometers, used for LOS path steps.
    - los_elevation_deg (float): Line-of-sight elevation angle in degrees from the horizon.
    - los_max_distance_km (float): Maximum distance along the line-of-sight to simulate.
    - alt_init (float): Initial altitude of the observer in meters.

    Outputs:
    ====================================================================
    - ds_final (xarray.Dataset): A new dataset containing the calculated line-of-sight wind
      and temperature data after Doppler processing.
    """

    # 1. Initial data preparation (VER calculation and vertical interpolation)
    
    # If 'Z3g' is not in the dataset (e.g., for WACCMX, or if TIEGCM wrapper failed),
    # calculate it from geopotential height ('Z3').
    if 'Z3g' not in ds:
        logger.info("'Z3g' not found in dataset. Attempting to calculate from 'Z3'.")
        ds = add_Z3g(ds, center_lat)

    # if Z3g is still not present or is all NaN, exit
    if 'Z3g' not in ds or ds['Z3g'].isnull().all():
        logger.warning(f"Z3g is missing or all NaN. Cannot perform vertical interpolation. Aborting processing for this location.")
        return xr.Dataset()
        
    ds = calculate_ver(ds)
    ds_vert_interp = interpolate_vertical(ds, grid_spacing_m)

    # check coordinates are in the range [-180, 180] for consistency
    ds_vert_interp = ds_vert_interp.assign_coords(lon=(((ds_vert_interp.lon + 180) % 360) - 180)).sortby('lon')

    # 2. Calculate all Line-of-Sight (LOS) points before interpolation
    los_calculator = LineOfSightCalculator(center_lat, center_lon, alt=alt_init)
    directions = ['N', 'E', 'S', 'W']
    los_points_by_dir = {} # for each cardinal direction, store the list of (lat, lon, alt) points
    all_lats, all_lons, all_alts = [], [], []

    for direction in directions:
        points = los_calculator.calculate_los(
            elevation_deg=los_elevation_deg,
            orientation=direction,
            step_km=grid_spacing_km,
            max_distance_km=los_max_distance_km
        )
        los_points_by_dir[direction] = points
        for p_lat, p_lon, p_alt in points:
            all_lats.append(p_lat)
            all_lons.append(p_lon)
            all_alts.append(p_alt)

    # 3. Define the scattered points for direct interpolation
    target_coords = {
        "lat": xr.DataArray(all_lats, dims="point"),
        "lon": xr.DataArray(all_lons, dims="point"),
        "altitude": xr.DataArray(all_alts, dims="point")
    }

    # 4. Perform a single interpolation from the vertically-gridded data directly to the scattered LOS points
    ds_los_interp = ds_vert_interp.interp(target_coords, method="linear", kwargs={'fill_value': np.nan})

    # 5. Init the DopplerProcessor with the new dataset and process the data
    doppler = DopplerProcessor(ds_los_interp, center_lat, center_lon, observer_alt=alt_init)
    los_data = doppler.process_doppler_optimized(los_points_by_dir, los_elevation_deg)

    # 6. Create the final output dataset coordinates
    times = ds['time'].values
    distance = np.arange(0, los_max_distance_km * 1000, grid_spacing_m)

    # Create an empty dataset to hold the final results
    ds_final = xr.Dataset(coords={'time': times, 'distance': distance})

    # populate the final dataset with all LOS variables
    ds_final = add_los_variables(ds_final, los_data, times, distance)

    return ds_final



def process_file(input_file, model_type, location, output_dir, grid_spacing_m, grid_spacing_km, los_elevation_deg, los_max_distance_km, alt_init):
    """Process the input file for a given location and save results.
    
    inputs:
    ====================================================================
    - input_file (str): Path to the input netCDF file.
    - model_type (str): Type of the model ('WACCMX' or 'TIEGCM').
    - location (str): Location code (e.g., 'A', 'B', 'C', etc.).
    - output_dir (str): Directory to save the output files.
    - grid_spacing_m (float): Grid spacing in meters for vertical interpolation and LOS steps.
    - grid_spacing_km (float): Grid spacing in kilometers for LOS steps.
    - los_elevation_deg (float): Line-of-sight elevation angle in degrees from the horizon.
    - los_max_distance_km (float): Maximum distance along the line-of-sight to simulate.
    - alt_init (float): Initial altitude of the observer in meters.
    
    outputs:
    ====================================================================
    - int: 0 on success, 1 on failure.
    """
    # make exception handling to ensure one failure doesn't stop the whole batch
    try:

        # Get location coordinates and names
        loc_coords, location_names = get_location_info()

        # Validate location
        if location not in loc_coords:
            raise ValueError(f"Invalid location code: {location}")

        # Extract location-specific parameters
        center_lat, center_lon = loc_coords[location]
        location_name = location_names[location]

        # show the processing info
        logger.info(f"Processing {input_file} for {location_name} (Model: {model_type})")

        # Open the raw dataset
        ds_raw = xr.open_dataset(input_file, engine='netcdf4')
        
        # use wrapper if TIEGCM
        if model_type == "TIEGCM":
            ds = tiegcm_to_waccmx_structure(ds_raw) # map variables and convert units
        else:
            # if waccmx - use as is
            ds = ds_raw
        
        # The rest of the pipeline now receives a dataset with the correct WACCMX structure
        ds_final = process_dataset(
            ds, center_lat, center_lon,
            grid_spacing_m=grid_spacing_m,
            grid_spacing_km=grid_spacing_km,
            los_elevation_deg=los_elevation_deg,
            los_max_distance_km=los_max_distance_km,
            alt_init=alt_init
        )

        # If the final dataset is empty (e.g., due to missing Z3g), skip saving
        output_path = generate_output_path(input_file, location_name, output_dir)
        ds_final.to_netcdf(output_path)
        logger.info(f"Saved results to {output_path}")

        return 0
    
    # log exceptions
    except Exception as e:
        logger.error(f"Failed to process {input_file} for {location}: {e}", exc_info=True)
        return 1
    




def main():
    """Main function to process the input file for all locations."""

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process WACCM-X or TIEGCM data for Doppler wind analysis.")
    parser.add_argument("input_file", type=str, help="Path to the input NetCDF file.")

    # self explanatory arguments with defaults
    parser.add_argument(
        "--model_type", type=str, default="WACCMX", choices=["WACCMX", "TIEGCM"],
        help="Type of the input model file. Use TIEGCM to activate the wrapper. Default: WACCMX."
    )
    parser.add_argument(
        "--locations", nargs='+', default=['kir', 'lyb', 'sod', 'caj', 'msh'],
        help="List of location codes to process (e.g., 'kir' 'lyb'). Default processes all five."
    )
    parser.add_argument(
        "--output_dir", type=str, default="/rds/projects/r/reidb-waccm-x/generated_files/therm_wind/",
        help="Base directory to save the output files."
    )
    parser.add_argument(
        "--grid_spacing_km", type=float, default=4/np.sqrt(2),
        help="Desired horizontal and vertical grid spacing in kilometers. Default: 4/sqrt(2)."
    )
    parser.add_argument(
        "--los_elevation_deg", type=float, default=45,
        help="Elevation angle for the line-of-sight (LOS) calculation in degrees from the horizon. Default: 45."
    )
    parser.add_argument(
        "--los_max_distance_km", type=float, default=600,
        help="Maximum distance for the line-of-sight (LOS) calculation in kilometers. Default: 600."
    )
    parser.add_argument(
        "--alt_init", type=float, default=50.0,
        help="Initial altitude of the observer in meters. Default: 50.0."
    )
    args = parser.parse_args()
    
    grid_spacing_m = args.grid_spacing_km * 1000
    
    logger.info(f"Starting processing for file: {args.input_file}")
    
    exit_code = 0
    

    # process the file if everything checks out
    for loc_code in args.locations:
        result = process_file(
            input_file=args.input_file,
            model_type=args.model_type, # Pass the model_type to the processing function
            location=loc_code,
            output_dir=args.output_dir,
            grid_spacing_m=grid_spacing_m,
            grid_spacing_km=args.grid_spacing_km,
            los_elevation_deg=args.los_elevation_deg,
            los_max_distance_km=args.los_max_distance_km,
            alt_init=args.alt_init
        )

        if result != 0:
            exit_code = 1
    
    return exit_code

# Run the main function if this script is executed
if __name__ == "__main__":
    sys.exit(main())