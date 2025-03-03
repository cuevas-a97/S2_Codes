#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## @package maint_computeDT
# @author Renaud Binet (CNES/DTN/TPI/QI)
# @version v1.0
# @date 2023
# @brief Compute the time difference between two bands of Sentinel 2
# @brief @b Example
# @brief <em>python main_computeDT.py -lon [longitude] -lat [latitude] -src [source_band] -dst [dest_band] -det [detector] -sat [satellite] -eph [ephemeris]</em>
# @param longitude Longitude
# @param latitude Latitude
# @param source_band First band
# @param dest_band Second band
# @param detector Detector corresponding to lat/lon point
# @param satellite Satellite A or B 
# @param ephemeris Ephemeris path file 
# @return Display the time difference between band source_band and dest_band


# Reference source 
#R. Binet, E. Bergsma, and V. Poulain. “ACCURATE SENTINEL-2 INTER-BAND TIME DELAYS.” ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences V-1-2022 (2022): 57–66. https://doi.org/10.5194/isprs-annals-V-1-2022-57-2022.


#-----------------------------------------------------------------------------------------------------------------------------
# IMPORT
#-----------------------------------------------------------------------------------------------------------------------------
from argparse import ArgumentParser
import xml.dom.minidom as minidom
import numpy as np
import os
import sys
import re

from proj_orbite import Calc_dt



#-----------------------------------------------------------------------------------------------------------------------------
# VARIABLES
#-----------------------------------------------------------------------------------------------------------------------------

# CSVREF informations
CSVREF_FILE = 'data_ref'
CSVREF_NAME = 'DT_REF_S2'
CSVREF_FORMAT = 'csv'

# radius of the earth
R = 6371000

# equatorial radius of the earth
R_eq = 6378137

# flattening 
f = 1./298.257223563

# time format for tile S2 L1C
time_format = "%Y-%m-%dT%H:%M:%S"



#-----------------------------------------------------------------------------------------------------------------------------
# FUNCTIONS
#-----------------------------------------------------------------------------------------------------------------------------


def readDatastripFile(eph):
    """
    Reads location data from the tile S2 L1C
    
    Input : 
        - eph[str]: ephemeris path file

    Output :
        - t_time[list]:   ephemeris informations acquisition time 
        - t_Xitrf[list]:  ephemeris informations x position
        - t_Yitrf[list]:  ephemeris informations y position
        - t_Zitrf[list]:  ephemeris informations z position
    """

    doc = minidom.parse(eph)
    gps=doc.getElementsByTagName('GPS_Points_List')[0]    
    t_Xitrf, t_Yitrf, t_Zitrf, t_time = [], [], [], []

    for i in range(1, len(gps.childNodes)-1, 2):
        xitrf, yitrf, zitrf = np.array(gps.childNodes[i].childNodes[1].childNodes[0].nodeValue.split(" ")).astype(np.float64)/1000
        time = gps.childNodes[i].childNodes[9].childNodes[0].nodeValue
        t_Xitrf.append(xitrf)
        t_Yitrf.append(yitrf)
        t_Zitrf.append(zitrf)
        t_time.append(time)
    return t_time, np.round(t_Xitrf, 3), np.round(t_Yitrf, 3), np.round(t_Zitrf, 3)
 

def check_args_conformity(args):
    """
    Checks the conformity of the arguments

    Input :
        - args[Parser.args]

    Output : void
    """
    
    # # src_band / dst_band   
    # if not re.match("^(B([1-9]\d*)|(B[0]{1}))(\.\d+)?$", args.src_band) or not re.match("^(B([1-9]\d*)|(B[0]{1}))(\.\d+)?$", args.dst_band):
    #     print("-src and -dst must match format B1, B2, ..., B13")
    #     print("Exemple : python sources/main_computeDT.py -lon 138.1145339 -lat -17.10045089 -src B1 -dst B2 -det D01 -sat B -eph data_validation/L1C_53KRB_20221208/DATASTRIP/DS_2BPS_20221208T015708_S20221208T005704/MTD_DS.xml")
    #     sys.exit(-1)

    # src_band / dst_band   
    if not re.match(r"^(B([1-9]\d*)|(B[0]{1}))(\.\d+)?$", args.src_band) or not re.match(r"^(B([1-9]\d*)|(B[0]{1}))(\.\d+)?$", args.dst_band):
        print("-src and -dst must match format B1, B2, ..., B13")
        print("Exemple : python sources/main_computeDT.py -lon 138.1145339 -lat -17.10045089 -src B1 -dst B2 -det D01 -sat B -eph data_validation/L1C_53KRB_20221208/DATASTRIP/DS_2BPS_20221208T015708_S20221208T005704/MTD_DS.xml")
        sys.exit(-1)
    
    # # detector
    # if not re.match("^D\d{2}?$", args.detector) :
    #     print("-det must match format D01, ..., D12")
    #     print("Exemple : python sources/main_computeDT.py -lon 138.1145339 -lat -17.10045089 -src B1 -dst B2 -det D01 -sat B -eph data_validation/L1C_53KRB_20221208/DATASTRIP/DS_2BPS_20221208T015708_S20221208T005704/MTD_DS.xml")
    #     sys.exit(-1)
    
    # detector
    if not re.match(r"^D\d{2}?$", args.detector):
        print("-det must match format D01, ..., D12")
        print("Exemple : python sources/main_computeDT.py -lon 138.1145339 -lat -17.10045089 -src B1 -dst B2 -det D01 -sat B -eph data_validation/L1C_53KRB_20221208/DATASTRIP/DS_2BPS_20221208T015708_S20221208T005704/MTD_DS.xml")
        sys.exit(-1)
    # # satellite
    # if args.satellite != "A" and args.satellite != "B":
    #     print("-sat must be \"A\" or \"B\"")
    #     print("Exemple : python sources/main_computeDT.py -lon 138.1145339 -lat -17.10045089 -src B1 -dst B2 -det D01 -sat B -eph data_validation/L1C_53KRB_20221208/DATASTRIP/DS_2BPS_20221208T015708_S20221208T005704/MTD_DS.xml")
    #     sys.exit(-1)

     # satellite
    if args.satellite != "A" and args.satellite != "B":
        print("-sat must be \"A\" or \"B\"")
        print("Exemple : python sources/main_computeDT.py -lon 138.1145339 -lat -17.10045089 -src B1 -dst B2 -det D01 -sat B -eph data_validation/L1C_53KRB_20221208/DATASTRIP/DS_2BPS_20221208T015708_S20221208T005704/MTD_DS.xml")
        sys.exit(-1)

#-----------------------------------------------------------------------------------------------------------------------------
# MAIN
#-----------------------------------------------------------------------------------------------------------------------------
  

if __name__ == "__main__":

    try:
        
        # Argument parser
        parser = ArgumentParser(description="Compute time difference between S2 bands.")
       
        parser.add_argument('-lon', '--longitude', type=float, help='longitude of L1C center point in degrees, ex: 1.683', required=True)
        parser.add_argument('-lat', '--latitude', type=float, help='latitude of L1C center point in degrees, ex: 7.643', required=True)
        parser.add_argument('-src', '--src_band', type=str, help='source band, such as B+N° in [1,12], ex: B1 or B2 or ... B12', required=True)
        parser.add_argument('-dst', '--dst_band', type=str, help='destination band, such as B+N° in [1,12], ex: B1 or B2 or ... B12', required=True)
        parser.add_argument('-det', '--detector', type=str, help='detector, such as D+N° in [01, 12], ex: D01 or D02 or ... D12', required=True)
        parser.add_argument('-sat', '--satellite', type=str, help='satellite, choose between A or B, ex: A', required=True)
        parser.add_argument('-eph', '--ephemFile', type=str, help='File containing Ephemeris, ephemeris xml path file', required=True)
        
        args = parser.parse_args()

        # Check args conformity
        check_args_conformity(args)

        # Update CSVREF_FILE path with satellite argument
        CSVREF_FILE = "{}/{}{}.{}".format(CSVREF_FILE, CSVREF_NAME, args.satellite, CSVREF_FORMAT)
     
        # Get Ephemerids from Datastrip 
        t_time, t_Xitrf, t_Yitrf, t_Zitrf = readDatastripFile(args.ephemFile)

        # Compute dt
        dt = Calc_dt(args.src_band, args.dst_band, args.detector, CSVREF_FILE, args.longitude, args.latitude, t_time, t_Xitrf, t_Yitrf, t_Zitrf, R, R_eq, f, time_format)
        print(dt)

        print("---- End ----")
        
    except Exception as e:
        
        print("Exception : {}".format(e))
        sys.exit(-1)
