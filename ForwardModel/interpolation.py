import numpy as np
import xarray as xr
import os
from scipy.interpolate import interp1d
from joblib import Parallel, delayed
from coord_transforms import geod, R_e, e2

# variables to interpolate
VARS_TO_INTERPOLATE = ['VER', 'U', 'V', 'T']

####### Vertical interpolation #######

def _interpolate_vertical_slice(t, z3g_data, vars_data, target_altitudes, var_list):
    """
    process one time slice from pre-loaded arrays.

    inputs:
    =======================================================================
    - t: time index
    - z3g_data: 4D array of Z3g data (time, altitude, latitude, longitude)
    - vars_data: dictionary of 4D arrays for each variable
    - target_altitudes: 1D array of altitudes to interpolate to
    - var_list: list of variable names to interpolate

    outputs:
    =======================================================================
    - interp_results: dictionary of interpolated arrays for each variable
    """

    # init the results dictionary
    interp_results = {}
    lat_dim = z3g_data.shape[2] # infer shapes from data to save loading lat/lon dimensions
    lon_dim = z3g_data.shape[3]

    # create empty arrays for each variable to interpolate
    for var in var_list:
        interp_results[var] = np.full((len(target_altitudes), lat_dim, lon_dim), np.nan)

    first_var_data = vars_data[var_list[0]] # just for shape and NaN mask

    # iterate over each latitude and longitude point
    for i in range(lat_dim):
        for j in range(lon_dim):

            # extract the column of data for the current latitude and longitude point
            z3g_col = z3g_data[t, :, i, j]
            first_var_col = first_var_data[t, :, i, j]

            valid = ~np.isnan(z3g_col) & ~np.isnan(first_var_col) # mask for valid data to supress NaNs for interpolation
            
            if np.sum(valid) < 2: # check if there are enough points to interpolate
                continue

            # Sort the altitude profile and get unique altitudes
            # This is needed to ensure interpolation works correctly
            alt_profile = z3g_col[valid]
            sort_indices = np.argsort(alt_profile) # increasing values only for interpolation
            alt_sorted = alt_profile[sort_indices]
            unique_alts, unique_indices = np.unique(alt_sorted, return_index=True)

            if len(unique_alts) < 2: # not enough unique altitudes to interpolate
                continue
            
            # Interpolate each variable to the target altitudes
            for var in var_list:

                var_profile = vars_data[var][t, :, i, j][valid][sort_indices] # apply mask and sort indices
                
                interp_func = interp1d(unique_alts, var_profile[unique_indices], kind='linear', bounds_error=False, fill_value=np.nan) # linear interpolation function
                interp_results[var][:, i, j] = interp_func(target_altitudes) # perform interpolation for the target altitudes
            
    return interp_results

def interpolate_vertical(ds, grid_spacing_m):
    """
    Performs vertical interpolation on all variables defined in VARS_TO_INTERPOLATE.

    inputs:
    =======================================================================
    - ds: xarray Dataset containing the variables to interpolate.
    - grid_spacing_m: Desired vertical spacing in metres.

    outputs:
    =======================================================================
    - interpolated_ds: xarray Dataset with interpolated variables.
    """

    print("Starting vertical interpolation...")

    #get the number of jobs from environment variable or default to 1
    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    print(f"Using {n_jobs} cores with shared-memory threading for vertical interpolation.")

    # Check if the dataset contains the required variable - make this configurable in future
    vars_in_ds = [var for var in VARS_TO_INTERPOLATE if var in ds.data_vars]

    # error if none of the variables are found
    if not vars_in_ds:
        raise ValueError("None of the variables specified in VARS_TO_INTERPOLATE were found in the dataset.")
    print(f"Interpolating variables: {vars_in_ds}")
    
    #get geometric altitudes
    z3g_data = ds['Z3g'].values

    vars_data = {var: ds[var].values for var in vars_in_ds} # select only the variables to interpolate
    
    # Use the grid_spacing_m parameter to create target altitudes
    target_altitudes = np.arange(np.nanmin(z3g_data), np.nanmax(z3g_data) + grid_spacing_m, grid_spacing_m)

    # perform vertical interpolation in parallel over time slices
    results = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(_interpolate_vertical_slice)(t, z3g_data, vars_data, target_altitudes, vars_in_ds) for t in range(len(ds.time))
    )

    # Combine results from all time slices into a single dataset
    final_arrays = {var: [] for var in vars_in_ds}
    for res_dict in results:
        for var in vars_in_ds:
            final_arrays[var].append(res_dict[var])
    
    # Convert lists to numpy arrays
    data_vars = {}
    for var in vars_in_ds:
        new_name = f"{var}_vertical_interp"
        full_array = np.array(final_arrays[var])

        # automate the naming process
        attrs = ds[var].attrs.copy()
        attrs['long_name'] = f"Interpolated {attrs.get('long_name', var)}"
        data_vars[new_name] = (('time', 'altitude', 'lat', 'lon'), full_array, attrs)
        
    # new dataset with interpolated values
    interpolated_ds = xr.Dataset(
        data_vars=data_vars,
        coords={'time': ds.time, 'altitude': target_altitudes, 'lat': ds.lat, 'lon': ds.lon}
    )
            
    print("Vertical interpolation finished.")
    return interpolated_ds

