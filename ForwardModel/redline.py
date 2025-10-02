import numpy as np
import xarray as xr
from constants import K_b_m  # Boltzmann constant in m^2 kg s^-2 K^-1

def calculate_ver(ds):
    """
    Calculate and add Volume Emission Rate (VER) at 630.0 nm to NetCDF file.
    
    Inputs:
    ====================================================================
    - ds : xarray.Dataset
    - file_path : str
        Path to the NetCDF file to process.
    - engine : str, optional
        NetCDF engine to use (default: 'netcdf4'). Options: 'netcdf4'.
    
    Outputs:
    ====================================================================
    - numpy.ndarray
        Calculated VER grid with shape (24, 126, 96, 144 for WACCMX).
    """
    
    Tn = ds['T'].values # K
    Ti = ds['TIon'].values # K
    n2_vmr = ds['N2_vmr'].values # N2 volume mixing ratio
    oplus_vmr = ds['Op'].values # O+ volume mixing ratio
    o2_vmr = ds['O2'].values # O2 volume mixing ratio
    o_vmr = ds['O'].values # O volume mixing ratio
    lev = ds['lev'].values # is already in hPa

    # Total density in m^-3, then cm^-3
    n_tot = lev[:, np.newaxis, np.newaxis] * 100 / (K_b_m * Tn)  # Pa / (J K⁻¹ * K) = m⁻³
    n_tot *= 1e-6  # Convert to cm^-3 for calculation
    n_n2 = n_tot * n2_vmr # N2 density in cm^-3
    n_OPlus = oplus_vmr * n_tot # O+ density in cm^-3
    O2_density = o2_vmr * n_tot # O2 density in cm^-3
    O_density = o_vmr * n_tot # O density in cm^-3

    #vectorized calculations for rate coefficients
    gamma_value = np.vectorize(gamma)(Tn, Ti)
    k1_value = np.vectorize(k1)(Tn)
    k2_value = np.vectorize(k2)(Tn)
    k3_value = np.vectorize(k3)(Tn)

    #calculate VER over entire grid
    grid_ver = VER(gamma_value, k1_value, k2_value, k3_value, A1n(), A2n(), O_density, O2_density, n_OPlus, n_n2)

    # add VER to the dataset as a DataArray
    ver_da = xr.DataArray(
        grid_ver,
        coords={'time': ds['time'], 'lev': ds['lev'], 'lat': ds['lat'], 'lon': ds['lon']},
        dims=('time', 'lev', 'lat', 'lon'),
        name='VER',
        attrs={'long_name': 'Volume Emission Rate at 630.0 nm', 'units': 'photons cm^-3 s^-1'}
    )
    return ds.assign(VER=ver_da)





def VER(gamma_value, k1_value, k2_value, k3_value, A1n, A2n, O, O2, OPlus, N2):
    """
    Calculate the volume emission rate (VER) of the 630.0 nm emission.

    densities are taken from WACCM-X

    gamma_value: rate coefficient for O+ + O2 -> O2+ + O
    k1_value: rate coefficient for O(1D) + N2 -> O + N2
    k2_value: rate coefficient for O(1D) + O2 -> O + O2
    k3_value: rate coefficient for O(1D) + O -> O + O
    A1n: rate coefficient for O(1D) -> O + hv(630.0 nm)
    A2n: rate coefficient for O(1D) -> O + hv(636.4 nm)
    O: concentration of O (cm^-3)
    O2: concentration of O2 (cm^-3)
    OPlus: concentration of O+ (cm^-3)
    N2: concentration of N2 (cm^-3)
    Returns: VER_630 (photons cm-3 s^-1)
    """

    ## from link and cogger 1988

    nom = (A1n * gamma_value * O2 * OPlus)
    den = (k1_value*N2 + k2_value*O2 + k3_value*O + A1n + A2n)
    
    return nom/den


# O+ + O2 -> O2+ + O (St. Maurice & Torr, 1978)
def gamma(Ti, Tn):
    """
    Rate coefficient for O+ + O2 -> O2+ + O.
    T_eff: effective temperature (K)
    Returns: gamma (cm^3 s^-1)
    """
    T_eff = (Ti + 2 * Tn) / 3  # Effective temperature for the reaction
    T_ratio = T_eff / 300
    return (2.82e-11 - 7.74e-12 * T_ratio + 1.07e-12 * T_ratio**2 -
            5.17e-14 * T_ratio**3 + 9.65e-16 * T_ratio**4)

# O(1D) + N2 -> O + N2 (Streit et al., 1976)
def k1(T_n):
    """
    Rate coefficient for O(1D) + N2 -> O + N2.
    T_n: neutral temperature (K)
    Returns: k1 (cm^3 s^-1)
    """

    # print(type(T_n), T_n.shape if hasattr(T_n, 'shape') else 'scalar')
    # print(f"T_n min: {np.min(T_n)}, max: {np.max(T_n)}, any negative: {np.any(T_n <= 0)}")
    # print(type(T_n))


    return 2e-11 * np.exp(107.8 / T_n)

#O(1D) + O2 -> O + O2 (Streit et al., 1976)
def k2(T_n):
    """
    Rate coefficient for O(1D) + O2 -> O + O2.
    T_n: neutral temperature (K)
    Returns: k2 (cm^3 s^-1)
    """
    return 2.9e-11 * np.exp(67.5 / T_n)

#  O(1D) + O -> O + O (Sun & Dalgarno, 1992)
def k3(T_n):
    """
    Rate coefficient for O(1D) + O -> O + O.
    T_n: neutral temperature (K)
    Returns: k3 (cm^3 s^-1)
    """
    return (3.73 + 1.1965e-1 * T_n**0.5 - 6.5898e-4 * T_n) * 1e-12

# O(1D) -> O + hv(630.0 nm) (Garstang, 1951)
def A1n():
    """
    Rate coefficient for O(1D) -> O + hv(630.0 nm).
    Returns: A1n (s^-1)
    """
    return 7.1e-3

# O(1D) -> O + hv(636.4 nm) (Garstang, 1951)
def A2n():
    """
    Rate coefficient for O(1D) -> O + hv(636.4 nm).
    Returns: A2n (s^-1)
    """
    return 2.2e-3