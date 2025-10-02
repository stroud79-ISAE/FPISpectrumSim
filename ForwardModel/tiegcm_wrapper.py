import xarray as xr
import numpy as np
import sys

# Import the necessary physical and atomic mass constants from your constants file.
from constants import K_b_m, m_oxygen, m_o2, m_n2

def tiegcm_to_waccmx_structure(ds_tiegcm):
    """
    Transforms a TIEGCM dataset to match the structure expected by the WACCMX processing scripts.
    """
    print("Applying TIEGCM wrapper to transform data structure...")
    
    if 'latlon' in ds_tiegcm.coords:
        ds_tiegcm = ds_tiegcm.drop_vars('latlon')

    ds_waccmx = xr.Dataset()

    # Coordinate Transformations
    ds_waccmx.coords['lat'] = ds_tiegcm['lat']
    ds_waccmx.coords['lon'] = ds_tiegcm['lon']
    ds_waccmx.coords['time'] = ds_tiegcm['time']

    p0_hpa = ds_tiegcm['p0'].item() # Reference pressure in hPa (millibars)
    pressure_hpa = p0_hpa * np.exp(-ds_tiegcm['lev']) # convert from dimensionless to hPa
    ds_waccmx.coords['lev'] = ('lev', pressure_hpa.data, {'units': 'hPa', 'long_name': 'pressure'})
    
    # defining target dimensions
    new_dims = ('time', 'lev', 'lat', 'lon')
    new_coords = {
        'time': ds_waccmx.coords['time'], 
        'lev': ds_waccmx.coords['lev'], 
        'lat': ds_waccmx.coords['lat'], 
        'lon': ds_waccmx.coords['lon']
    }
    
    #ensure variables follow this structure for coordinates
    ds_waccmx['U'] = xr.DataArray(ds_tiegcm['UN'].values / 100.0, dims=new_dims, coords=new_coords)
    ds_waccmx['U'].attrs = {'units': 'm/s', 'long_name': 'Zonal wind'}

    ds_waccmx['V'] = xr.DataArray(ds_tiegcm['VN'].values / 100.0, dims=new_dims, coords=new_coords)
    ds_waccmx['V'].attrs = {'units': 'm/s', 'long_name': 'Meridional wind'}

    ds_waccmx['T'] = xr.DataArray(ds_tiegcm['TN'].values, dims=new_dims, coords=new_coords)
    ds_waccmx['T'].attrs = {'units': 'K', 'long_name': 'Temperature'}

    ds_waccmx['TIon'] = xr.DataArray(ds_tiegcm['TI'].values, dims=new_dims, coords=new_coords)
    ds_waccmx['TIon'].attrs = {'units': 'K', 'long_name': 'Ion Temperature'}


    # if ilev is a coordinate in geopotential height - swap it to lev
    if 'ilev' in ds_tiegcm['Z'].dims:
        # If 'Z' has 'ilev', swap it to 'lev' first
        Z_data = ds_tiegcm['Z'].swap_dims({'ilev': 'lev'})
        ds_waccmx['Z3'] = xr.DataArray(Z_data.values / 100.0, dims=new_dims, coords=new_coords)
    else:
        ds_waccmx['Z3'] = xr.DataArray(ds_tiegcm['Z'].values / 100.0, dims=new_dims, coords=new_coords)
    ds_waccmx['Z3'].attrs = {'units': 'm', 'long_name': 'Geopotential Height'}

    # --- NEW LOGIC ---
    # Map ZG (geometric height in km) to Z3g (geometric height in m) if it exists
    if 'ZG' in ds_tiegcm:
        print("Found 'ZG' in TIEGCM file, mapping directly to 'Z3g'.")
        if 'ilev' in ds_tiegcm['ZG'].dims:
            zg_data = ds_tiegcm['ZG'].swap_dims({'ilev': 'lev'})
        else:
            zg_data = ds_tiegcm['ZG']
        
        # Convert ZG from km to meters
        ds_waccmx['Z3g'] = xr.DataArray(zg_data.values * 1000.0, dims=new_dims, coords=new_coords)
        ds_waccmx['Z3g'].attrs = {'units': 'm', 'long_name': 'Geometric Height'}
    else:
        print("'ZG' not found in TIEGCM file. 'Z3g' will need to be calculated from 'Z3'.")

#####################################################################################################
    # Variables for VER Calculation
    pressure_pa = ds_waccmx.coords['lev'] * 100 # pressure in Pa
    n_tot_m3 = pressure_pa / (K_b_m * ds_waccmx['T']) # total density 
    n_tot_m3.attrs = {'units': 'm^-3', 'long_name': 'Total number density'}
    n_tot_cm3 = n_tot_m3 * 1e-6 # cm^-3 conversion

    den_kg_m3 = xr.DataArray(ds_tiegcm['DEN'].values * 1000.0, dims=new_dims, coords=new_coords)


    # Calculate Op VMR
    op_density_cm3 = xr.DataArray(ds_tiegcm['OP'].values, dims=new_dims, coords=new_coords)
    op_vmr = op_density_cm3 / n_tot_cm3
    ds_waccmx['Op'] = op_vmr
    ds_waccmx['Op'].attrs = {'units': 'mol/mol', 'long_name': 'Op concentration'}

    # O VMR
    mass_density_o = xr.DataArray(ds_tiegcm['O1'].values, dims=new_dims, coords=new_coords) * den_kg_m3
    number_density_o = mass_density_o / m_oxygen
    o_vmr = number_density_o / n_tot_m3
    ds_waccmx['O'] = o_vmr
    ds_waccmx['O'].attrs = {'units': 'mol/mol', 'long_name': 'O concentration'}

    # O2 VMR
    mass_density_o2 = xr.DataArray(ds_tiegcm['O2'].values, dims=new_dims, coords=new_coords) * den_kg_m3
    number_density_o2 = mass_density_o2 / m_o2
    o2_vmr = number_density_o2 / n_tot_m3
    ds_waccmx['O2'] = o2_vmr
    ds_waccmx['O2'].attrs = {'units': 'mol/mol', 'long_name': 'O2 concentration'}
    
    # N2 VMR
    n2_density = xr.DataArray(ds_tiegcm['N2'].values, dims=new_dims, coords=new_coords) * den_kg_m3
    number_density_n2 = n2_density / m_n2
    n2_vmr = number_density_n2 / n_tot_m3
    ds_waccmx['N2_vmr'] = n2_vmr
    ds_waccmx['N2_vmr'].attrs = {'units': 'mol/mol', 'long_name': 'N2 concentration'}

#####################################################################################################

    # clean up 
    for var in ds_waccmx.data_vars:
        if 'coordinates' in ds_waccmx[var].attrs:
            del ds_waccmx[var].attrs['coordinates']
        ds_waccmx[var] = ds_waccmx[var].transpose('time', 'lev', 'lat', 'lon', ...)

    print("TIEGCM data successfully transformed to WACCMX-like structure.")
    return ds_waccmx
