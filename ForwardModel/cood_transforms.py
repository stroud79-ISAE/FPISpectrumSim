from pyproj import Geod
import numpy as np

## Earth geometry
geod = Geod(ellps='WGS84')
R_e = 6378137.0  # Semi-major axis (m)
f = 1.0 / 298.257223563 # Flatting factor
e2 = (2.0 * f) - (f**2) # Eccentricity squared



##################################################
####### Coordinate tranformation functions #######
##################################################

def wgs2ecef(lat, lon, alt):
    """
    Convert WGS84 geodetic coords to ECEF Cartesian coords.
    Input: lat, lon (deg), alt (km)
    Output: x, y, z (m)
    """
    
    rlat = np.deg2rad(lat)
    rlon = np.deg2rad(lon)
    ralt = 1e3 * alt  # km to m
    sa = np.sin(rlat)
    ca = np.cos(rlat)
    N = R_e / np.sqrt(1.0 - e2 * (sa**2))
    x = (N + ralt) * ca * np.cos(rlon)
    y = (N + ralt) * ca * np.sin(rlon)
    z = ((1.0 - e2) * N + ralt) * sa
    return x, y, z



def ecef2wgs(x, y, z, tol=1.0e-6):
    """
    Convert from ECEF cartesian coordinates to WGS ellipsoidal coords.
    ECEF = [x, y, z], [m, m, m]
    WGS = [lat, lon, alt], [deg, deg, km]
    """
    
    rad, rlat, rlon = cartsph(x, y, z)
    p = np.sqrt((x**2) + (y**2))
    zp = z / p
    la = np.arctan2(zp, 1.0 - (e2))
    la0 = rlat + 1.0e3
    h = rad
    zn = np.abs(z) - (1.0 - e2) * R_e / np.sqrt(1.0 - e2)
    for i in range(100):
        la0 = la + 0.0
        N = R_e / (np.sqrt(1.0 - e2 * (np.sin(la) ** 2)))
        h = np.fmax((p / np.cos(la)) - N, zn)
        la = np.arctan2(zp, 1.0 - ((N / (N + h)) * e2))
        if np.all(np.abs(la0 - la) < tol):
            break
    lat = np.rad2deg(la)
    lon = np.rad2deg(rlon)
    alt = 1.0e-3 * h
    
    
    return lat, lon, alt




def cartsph(x, y, z):
    """
    Convert from ECEF cartesian coordinates to spherical coords.
    ECEF = [x, y, z], [m, m, m]
    Sph = [radius, lat, lon], [m, rad, rad]
    """
    rad = np.sqrt(x**2 + y**2 + z**2)
    lat = np.arctan2(z, np.sqrt(x**2 + y**2))
    lon = np.arctan2(y, x)
    return rad, lat, lon



def enu2xyz(E, N, U, lat, lon):
    """
    Convert East-North-Up to ECEF X-Y-Z.
    ENU = [E, N, U], [m, m, m]
    ECEF = [X, Y, Z], [m, m, m]
    lat, lon in degrees.
    """
    rlat = np.deg2rad(lat)
    rlon = np.deg2rad(lon)
    sa = np.sin(rlat)
    ca = np.cos(rlat)
    so = np.sin(rlon)
    co = np.cos(rlon)
    X = -so * E - co * sa * N + co * ca * U
    Y = co * E - so * sa * N + so * ca * U
    Z = ca * N + sa * U
    return X, Y, Z