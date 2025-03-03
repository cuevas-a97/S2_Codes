# coding: utf-8
'''

@Author :  Renaud Binet (CNES/DTN/TPI/QI)
@Date : 2023

'''

#Reference source
#R. Binet, E. Bergsma, and V. Poulain. “ACCURATE SENTINEL-2 INTER-BAND TIME DELAYS.” ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences V-1-2022 (2022): 57–66. https://doi.org/10.5194/isprs-annals-V-1-2022-57-2022.


from datetime import datetime
import numpy as np
import math
import os
import sys
import csv


def distance_lon_lat(lat1, lon1, lat2, lon2, R):
    """
    Returns the distance between 2 points on spherical earth: suitable for long distances (>10km)
    
    Input:
        @param lat1[float]: latitude of the first point in radians
        @param lon1[float]: longitude of the first point in radians
        @param lat2[float]: latitude of the second point in radians
        @param lon2[float]: longitude of the second point in radians
        @param R[int]: radius of the earth in meters

    Output:
        - d[float]: the distance between 2 points on spherical earth in meters
    """

    f1 = lat1
    f2 = lat2
    dlat = (lat2 - lat1)
    dlon = (lon2 - lon1)
    a = pow(math.sin(dlat / 2), 2) + math.cos(f1) * math.cos(f2) * pow(math.sin(dlon / 2), 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = R * c

    return d


def distance_lon_lat_ellipsoid(lat1, lon1, lat2, lon2, R_eq, f):
    """
    Returns the distance between 2 points on ellipsoidal earth, suitable for short distances (<10km) because local flat earth approximation
      
    Input:
        @param lat1[float]: latitude of the first point in radians
        @param lon1[float]: longitude of the first point in radians
        @param lat2[float]: latitude of the second point in radians
        @param lon2[float]: longitude of the second point in radians
        @param R_eq[int]: equatorial radius of the earth in meters
             
    Output:
        - d[float]: the distance between 2 points on ellipsoidal earth in meters
    """
    
    e2 = f * (2.-f)

    # Local curvature radius
    R1 = R_eq * (1.-e2) / pow((1 - e2 * pow(math.sin(lat1), 2)), 1.5)  
    R2 = R_eq / math.sqrt(1. - e2 * pow(math.sin(lat1), 2))
    
    f1 = lat1
    f2 = lat2
    dlat = (lat2 - lat1)
    dlon = (lon2 - lon1)
    distance_North = R1 * dlat
    distance_East = R2 * math.cos(lat1) * dlon
    d = math.sqrt(distance_North * distance_North + distance_East * distance_East)

    return d

def ITRF_to_latlon(x, y, z, R_eq, f):
    """
    Computes lat, long, height/ellipsoid of an ITRF x, y, z map coordinate
    Accuracy better than 1cm

    Input : 
        @param x[float]: x ITRF coordinate
        @param y[float]: y ITRF coordinate
        @param z[float]: z ITRF coordinate
        @param R_eq[int]: equatorial radius of the earth in meters
        @param f[float]: flattening

    Output:
        - lat[float | list]: latitude of x, y, z in radians
        - lon[float | list]: longitude of x, y, z in radians
        - h[float | list]: height of x, y, z in meters
    """

    e2 = f * (2. - f)
    eps = e2 / (1. - e2)
    b = R_eq * (1. - f)
        
    p = math.sqrt(x * x + y * y)
    q = math.atan2(z * R_eq, p * b)
    sin3q = pow(math.sin(q), 3)
    cos3q = pow(math.cos(q), 3)
    lat = math.atan2(z + (eps * b * sin3q), p - (e2 * R_eq * cos3q))  
    lon = math.atan2(y, x)
    v = R_eq / math.sqrt(1 - (e2 * pow(math.sin(lat), 2)))
    h = p / math.cos(lat) - v   

    return lat, lon, h

def GPS_to_Vground(time, x, y, z, R_eq, f, time_format):
    """
    Returns the velocity of the (x,y,z)/ITRF orbital ephemeris ground track, as well as lat, long, height/ellipsoid
    Velocity is obtained by non-centered differentiation

    Input : 
        @param time[list]: acquisition times
        @param x[list]: x ITRF coordinates
        @param y[list]: y ITRF coordinates
        @param z[list]: z ITRF coordinates
        @param R_eq[int]: equatorial radius of the earth in meters
        @param f[float]: flattening
        @param time_format[str]: time format for datetime

    Output :
        - v[list]: velocity of the (x,y,z)/ITRF orbital ephemeris ground track in m/s
        - h[list]: height of the (x,y,z)/ITRF orbital ephemeris ground track in meters
        - lat[list]: latitude of the (x,y,z)/ITRF orbital ephemeris ground track in radians
        - lon[list]: longitude of the (x,y,z)/ITRF orbital ephemeris ground track in radians
        
        Returned lists are shorter by one element: the last element is not an evalue
    """

    v, h, lat, lon = [], [], [], []
    
    for index,tmp in enumerate(x[0:-1]):
        lat1, lon1, h1 = ITRF_to_latlon(x[index], y[index], z[index], R_eq, f)
        lat2, lon2, h2 = ITRF_to_latlon(x[index+1], y[index+1], z[index+1], R_eq, f)
        delta_time = (datetime.strptime(time[index + 1], time_format) - datetime.strptime(time[index], time_format)).total_seconds()
        v.append(distance_lon_lat_ellipsoid(lat1, lon1, lat2, lon2, R_eq, f) / delta_time)
        h.append(h1)
        lat.append(lat1)
        lon.append(lon1)

    return v, h, lat, lon

def nearest_ephemeris(lon, lat, t_time, t_Xitrf, t_Yitrf, t_Zitrf, R, R_eq, f, time_format):
    """
    Computes the nearest ephemeris to the center of the tile S2 L1C lon, lat

    Input : 
        @param lon[float]: longitude of the center of the tile S2 L1C in degrees
        @param lat[float]: latitude of the center of the tile S2 L1C in degrees
        @param t_time[lst]: acquisition times from orbital ephemeris in seconds
        @param t_Xitrf[lst]: x ITRF positions from orbital ephemeris
        @param t_Yitrf[lst]: y ITRF positions from orbital ephemeris
        @param t_Zitrf[lst]: z ITRF positions from orbital ephemeris
        @param R[int]: radius of the earth in meters
        @param R_eq[int]: equatorial radius of the earth
        @param f[float]: flattening
        @param time_format[str]: time format for datetime

    Output : 
        - Hsat[float]: height of the nearest ephemeris in meters
        - Vground[float]: velocity of the nearest ephemeris in m/s
    """

    
    # Computation of Hsat, Vground, lon, lat for each GPS ephemeris
    Vground, Hsat, lat_GPS, lon_GPS = GPS_to_Vground(t_time, t_Xitrf, t_Yitrf, t_Zitrf, R_eq, f, time_format)
    lat_GPS = np.array(lat_GPS)
    lon_GPS = np.array(lon_GPS)
    Ngps = len(lat_GPS)
    
    # Association of Vground and Hsat for each point : we take the Vground (resp. Hsat) of the closest ephemeris in lon/lat
    # Computation of distances to each ephemeris
    t_dist = [distance_lon_lat(lat / 180. * np.pi, lon / 180. * np.pi, lat_GPS, lon_GPS, R) for (lat_GPS, lon_GPS) in zip(lat_GPS, lon_GPS)]
    index_min = (np.argmin(t_dist))

    return Hsat[index_min], Vground[index_min]

def Calc_dt(src_band, dst_band, detector, s_csv_ref, lon, lat, t_time, t_Xitrf, t_Yitrf, t_Zitrf, R, R_eq, f, time_format):
    """
    Returns the estimated dt for a pair of bands / detector from a reference csv file df_ref and orbital ephemerides t_time, t_Xitrf, t_Yitrf, t_Zitrf. t_time in seconds in the day
    ex: src_band='B2', dst_band='B4', detector='D01'

    Input : 
        @param src_band[str]: source band
        @param dst_band[str]: destination band
        @param detector[str]: detector
        @param s_csv_ref[str]: file path for the csv reference
        @param lon[float]: longitude from the center of the tile S2 L1C in decimal degrees
        @param lat[float]: latitude from the center of the tile S2 L1C in decimal degrees
        @param t_time[lst]: acquisition times from orbital ephemeris in seconds
        @param t_Xitrf[lst]: x ITRF positions from orbital ephemeris
        @param t_Yitrf[lst]: y ITRF positions from orbital ephemeris
        @param t_Zitrf[lst]: z ITRF positions from orbital ephemeris
        @param R[int]: radius of the earth
        @param R_eq[int]: equatorial radius of the earth
        @param f[float]: flattening
        @param time_format[str]: time format for datetime


    Output : 
        - dt[float]: estimated dt in sec
    """
    
    # Computation of satellite altitude and gound velocity for the point considered
    Hsat, Vground = nearest_ephemeris(lon, lat, t_time, t_Xitrf, t_Yitrf, t_Zitrf, R, R_eq, f, time_format)
    
    # Retrieval of the reference table which contains for each pair of bands/detectors the reference delta_t dt_ref, as well as the altitude and Vground associated to the dt_ref measurement
    dt_ref, Hsat_ref, Vground_ref = -1, -1, -1
    with open(s_csv_ref) as csvfile :
        csv_reader = csv.reader(csvfile, delimiter=',')
        header = next(csv_reader)
        for row in csv_reader:
            if row[0] == src_band and row[1] == dst_band and row[2] == detector :
                dt_ref = float(row[3])
                Hsat_ref = float(row[4])
                Vground_ref = float(row[5])
    
#     Verification of the correct reading of the csv
    if dt_ref == -1 or Hsat_ref == -1 or Vground_ref == -1:
        print("Error when reading csv file : dt_ref or Hsat_ref or Vground_ref")
        sys.exit(-1)
    
    dt = dt_ref * Hsat / Hsat_ref * Vground_ref / Vground

    return dt
