import math
import numpy as np
import xarray as xr
from joblib import Parallel, delayed
import os
from coord_transforms import wgs2ecef, ecef2wgs, enu2xyz, geod
from constants import c, f_0, m_oxygen, K_b_m



####### Line of sight calculator ########
class LineOfSightCalculator:
    def __init__(self, lat, lon, alt):
        self.start_lat = lat
        self.start_lon = lon
        self.start_alt = alt

    def calculate_los(self, elevation_deg, orientation, step_km=4/math.sqrt(2), max_distance_km=1000):
        '''
        Calculate line of sight points based on elevation and orientation.

        Inputs:
            elevation_deg : elevation in degrees
            orientation : looking direction ('N', 'S', 'E', 'W')
            step_km : sample rate along line of sight in km - default 2.828...km
            max_distance_km : maximum distance to calculate to in km - default 1000km

        Outputs:
            points : list of tuples (lat, lon, alt) for each point along the line of sight

        '''

        ## conversions
        elevation_rad = math.radians(elevation_deg) # convert elevation to radians
        bearing_rad = math.radians(self._orientation_to_bearing(orientation)) # convert orientation to radians
        step_m = step_km * 1000 # convert step to meters
        max_distance_m = max_distance_km * 1000 # convert max distance to meters

        # init points list
        points = []
        x0, y0, z0 = wgs2ecef(self.start_lat, self.start_lon, self.start_alt * 1e-3)

        # calculate direction vector in ECEF coordinates
        cos_elevation = math.cos(elevation_rad)
        sin_elevation = math.sin(elevation_rad)
        cos_bearing = math.cos(bearing_rad)
        sin_bearing = math.sin(bearing_rad)

        east = cos_elevation * sin_bearing
        north = cos_elevation * cos_bearing
        up = sin_elevation

        dir_x, dir_y, dir_z = enu2xyz(east, north, up, self.start_lat, self.start_lon)
        dir_ecef = np.array([dir_x, dir_y, dir_z])
        dir_ecef = dir_ecef / np.sqrt(sum(x * x for x in dir_ecef))

        # calculate points along the line of sight
        num_points = int(np.floor((max_distance_m + step_m) / step_m))
        distances = np.arange(0, num_points * step_m, step_m)

        for d in distances:
            x = x0 + d * dir_ecef[0]
            y = y0 + d * dir_ecef[1]
            z = z0 + d * dir_ecef[2]
            lat, lon, alt = ecef2wgs(x, y, z)
            alt_m = alt * 1000
            points.append((lat, lon, alt_m))

        return points


    def _orientation_to_bearing(self, orientation):
        '''helper function to convert orientation to bearing in degrees'''
        orientation = orientation.upper()
        bearings = {'N': 0, 'E': 90, 'S': 180, 'W': 270}
        if orientation not in bearings:
            raise ValueError(f"Invalid orientation '{orientation}'. Use N, E, S, or W.")
        return bearings[orientation]

####### Doppler spectrum class #######
class DopplerProcessor:
    def __init__(self, ds_los_interp, observer_lat, observer_lon, observer_alt=50):
        self.ds_los_interp = ds_los_interp
        self.observer_lat = observer_lat
        self.observer_lon = observer_lon
        self.observer_alt = observer_alt
        self.times = self.ds_los_interp.time.values
        self.directions = {'N': 0, 'E': 90, 'W': 270, 'S': 180}

    def _process_task(self, task, los_points_by_dir, elevation_deg, point_offsets):
        """
        Helper function to process a single direction for a single time step.

        This is for the parallelisation task.

        Inputs: 
            task : tuple (time_index, direction)
            los_points_by_dir : dictionary of line of sight points by direction
            elevation_deg : elevation angle in degrees
            point_offsets : pre-calculated start and end indices for each direction's data points
        Outputs:
            tuple of (time_index, direction, data) where data is a dictionary containing:
                - 'v_los': line-of-sight wind speed
                - 'T': temperature
                - 'emission_rates': vertical emission rates
                - 'wind_doppler': integrated Doppler wind speed
                - 'temp_doppler': integrated Doppler temperature
                - 'spectrum': normalized Doppler spectrum
                - 'f_range': frequency range for the spectrum
        """

        
        t, direction = task # time index and direction
        bearing = self.directions[direction] 

        # Select data for the current time step
        ds_t = self.ds_los_interp.isel(time=t)

        # Get the pre-calculated start and end indices for this direction's points
        start_idx, end_idx = point_offsets[direction]
        
        # Select the slice of points corresponding to the current direction
        data_for_dir = ds_t.isel(point=slice(start_idx, end_idx))
        
        # Extract values
        u_values = data_for_dir['U_vertical_interp'].values
        v_values = data_for_dir['V_vertical_interp'].values
        t_values = data_for_dir['T_vertical_interp'].values
        ver_values = data_for_dir['VER_vertical_interp'].values

        #calculations
        v_los = self.line_of_sight_wind(u_values, v_values, theta=elevation_deg, phi=bearing)
        f_mean, sigma_f, spectrum, f_range = self.doppler_spectrum(v_los, t_values, ver_values)
        wind_d = self.wind_speed_from_doppler(f_mean, f_0)
        temp_d = self.temperature_from_doppler(sigma_f, f_0)

        # results in dict structure
        return (t, direction, {
            'v_los': v_los, 'T': t_values, 'emission_rates': ver_values,
            'wind_doppler': wind_d, 'temp_doppler': temp_d,
            'spectrum': spectrum, 'f_range': f_range
        })

    def process_doppler(self, los_points_by_dir, elevation_deg):
        """
        processing method that parallelies over both time and direction.
        """
        n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
        print(f"Using {n_jobs} cores for Doppler processing (parallelized over time and direction).")

        # list of all (time, direction) tasks to be processed.
        tasks = []
        for t in range(len(self.times)):
            for direction in self.directions:
                tasks.append((t, direction))

        # pre-calculate the start/end index for each direction's data points
        point_offsets = {}
        current_offset = 0
        for d in self.directions:
            num_points = len(los_points_by_dir[d])
            point_offsets[d] = (current_offset, current_offset + num_points)
            current_offset += num_points

        # run processing in parallel for each (time, direction) task.
        all_results = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(self._process_task)(task, los_points_by_dir, elevation_deg, point_offsets) for task in tasks
        )

        # reconstruct the final dict from the flat list of results.
        output_keys = ['v_los', 'T', 'emission_rates', 'wind_doppler', 'temp_doppler', 'spectrum']
        los_data = {
            direction: {key: [None] * len(self.times) for key in output_keys}
            for direction in self.directions
        }
        f_range_final = None

        # Populate the structure with the results from the parallel jobs.
        for t, direction, data in all_results:
            for key, value in data.items():
                if key == 'f_range':
                    if f_range_final is None:
                        f_range_final = value
                else:
                    los_data[direction][key][t] = value
        
        # Convert lists to numpy arrays for the final output.
        for direction in self.directions:
            for key, value in los_data[direction].items():
                los_data[direction][key] = np.array(value)
            # Assign the single frequency range to each direction's data.
            los_data[direction]['f_range'] = f_range_final

        return los_data

    def line_of_sight_wind(self, u, v, theta, phi):
        '''
        Calculate the line-of-sight wind speed based on the wind components and angles.
        '''
        theta_rad = np.radians(theta)
        phi_rad = np.radians(phi)
        los_x = np.sin(theta_rad) * np.sin(phi_rad)
        los_y = np.sin(theta_rad) * np.cos(phi_rad)
        v_los = u * los_x + v * los_y
        return v_los

    def doppler_spectrum(self, v_los, T, emission_rates):
        '''
        Calculate the Doppler spectrum based on the line-of-sight wind speed, temperature, and emission rates.

        Inputs:
            v_los : 
        '''
        # init arrays
        v_los = np.asarray(v_los)
        T = np.asarray(T)
        emission_rates = np.asarray(emission_rates)

        f_p = f_0 * (c + v_los) / c # Doppler shifted frequency
        sigma_fp = np.sqrt(K_b_m * T / (m_oxygen * c**2)) * f_0 # Doppler broadening

        # If all sigma_fp are NaN, return NaN values
        if np.all(np.isnan(sigma_fp)):
            return np.nan, np.nan, np.full(1000, np.nan), np.linspace(f_0-1e9, f_0+1e9, 1000)

        max_sigma_fp = np.nanmax(sigma_fp) # maximum Doppler broadening
        f_center = f_0 # center frequency
        f_range = np.linspace(f_center - 5 * max_sigma_fp, f_center + 5 * max_sigma_fp, 1000) # range of frequencies to map doppler spectrum to
        df = f_range[1] - f_range[0] # df
        spectrum_total = np.zeros_like(f_range) # init spectrum

        # Calculate the spectrum for each point
        for i in range(len(v_los)):

            # If the values are not NaN, calculate the spectrum
            if not np.isnan(v_los[i]) and not np.isnan(T[i]) and not np.isnan(emission_rates[i]):
                spectrum = emission_rates[i] * (1 / (np.sqrt(2 * np.pi) * sigma_fp[i])) * \
                           np.exp(-((f_range - f_p[i])**2) / (2 * sigma_fp[i]**2)) # maxwellian distribution
                spectrum_total += spectrum # append each element
        integral = np.sum(spectrum_total * df) # integral of the spectrum

        # Normalize the spectrum if the integral is not zero (greater that epsilon)
        if integral > 1e-10:
            spectrum_total /= integral

        # Calculate the mean frequency and standard deviation
        f_mean = np.sum(f_range * spectrum_total * df)
        sigma_f = np.sqrt(np.sum((f_range - f_mean)**2 * spectrum_total * df))

        return f_mean, sigma_f, spectrum_total, f_range

    def wind_speed_from_doppler(self, f_mean, f_0):
        '''
        Calculate the wind speed from the Doppler frequency shift.
        '''
        return c * (f_mean / f_0 - 1)

    def temperature_from_doppler(self, sigma_f, f_0):
        '''
        Calculate the temperature from the Doppler broadening frequency.
        '''
        return (m_oxygen * c**2 * sigma_f**2) / (K_b_m * f_0**2)