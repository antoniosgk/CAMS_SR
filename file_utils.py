'''
This file_utils.py is used in order to construct the paths of the stations file
the species,the temperature and the pressure
'''
import pathlib
stations_path = "/home/agkiokas/CAMS/CHINESE_STATIONS_INFO_2015_2023.txt"  # your stations table
base_path = "/mnt/store01/agkiokas/CAMS/inst"

# species and dataset naming convention
product = "inst3d"     # filename prefix,depends on the name of the file saved
species = "O3"          # e.g. "O3", "CO2", ...
date = "20050524"       # YYYYMMDD
time = "0200"           # HHMM

# Path construction for species file 
species_file = pathlib.Path(f"{base_path}/{species}/{product}_{date}_{time}.nc4")

# Path construction for Pressure Level file 
pl_file = pathlib.Path(f"{base_path}/PL/{product}_{date}_{time}.nc4")

# Path construction for Temperature file
T_file = pathlib.Path(f"{base_path}/T/{product}_{date}_{time}.nc4")
