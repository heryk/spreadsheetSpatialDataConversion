"""
Module: spatial_converter

This module provides functions to convert geographic coordinates 
from various formats to Decimal Degrees (DD).

Functions:
- convert_dms_to_dd(row): Converts Degree/Minute/Second (DMS) coordinates to Decimal Degrees (DD).
- convert_utm_to_dd(row): Converts Universal Transverse 
  Mercator (UTM) coordinates to Decimal Degrees (DD).

Dependencies:
- pandas: Used for data manipulation and handling.
- pyproj: Used for coordinate transformation.
- pathlib: Used for file path operations.

Usage:
This module is intended to be used as part of a larger data processing pipeline for geographic data. 
The functions take rows of data as input and return the converted coordinates in Decimal Degrees.

Example:
    import pandas as pd
    from spatial_converter import convert_dms_to_dd, convert_utm_to_dd

    # Example DataFrame with DMS coordinates
    df = pd.DataFrame({
        'deglat': [40, 50],
        'minlat': [30, 45],
        'seclat': [0, 30],
        'deglong': [-75, -80],
        'minlong': [15, 30],
        'seclong': [0, 45]
    })

    # Convert DMS to DD
    df[['lat_decimal', 'lon_decimal']] = df.apply(convert_dms_to_dd, axis=1)

    # Example DataFrame with UTM coordinates
    df_utm = pd.DataFrame({
        'utmeast': [500000, 400000],
        'utmnorth': [4649776, 5000000],
        'utmsrid': [32633, 32634]
    })

    # Convert UTM to DD
    df_utm[['lat_decimal', 'lon_decimal']] = df_utm.apply(convert_utm_to_dd, axis=1)
"""
from pathlib import Path
import pandas as pd
import pyproj
from pyproj import Transformer
# import numpy as np


def convert_dms_to_dd(row):
    """Convert Degree/Minute/Second coordinates to Decimal Degrees"""
    if pd.notna(row['deglat']) and pd.notna(row['minlat']) and pd.notna(row['seclat']):
        lat = row['deglat'] + (row['minlat'] / 60) + (row['seclat'] / 3600)
    else:
        lat = None

    if pd.notna(row['deglong']) and pd.notna(row['minlong']) and pd.notna(row['seclong']):
        lon = row['deglong'] + (row['minlong'] / 60) + (row['seclong'] / 3600)
    else:
        lon = None

    return pd.Series({'lat_decimal': lat, 'lon_decimal': lon})

def convert_utm_to_dd(row):
    """Convert UTM coordinates to Decimal Degrees"""
    if pd.notna(row['utmeast']) and pd.notna(row['utmnorth']) and pd.notna(row['utmsrid']):
        # Create transformer from UTM to WGS84
        utm_proj = pyproj.CRS.from_epsg(row['utmsrid'])
        wgs84 = pyproj.CRS.from_epsg(4326)
        transformer = Transformer.from_crs(utm_proj, wgs84, always_xy=True)

        # Transform coordinates
        lon, lat = transformer.transform(row['utmeast'], row['utmnorth'])
        return pd.Series({'lat_decimal': lat, 'lon_decimal': lon})
    else:
        return pd.Series({'lat_decimal': None, 'lon_decimal': None})

def process_spatial_data(local_input_file, local_output_file):
    """
    Process spatial data from input CSV file and convert coordinates to decimal degrees
    """
    try:
        # Read the CSV file
        df = pd.read_csv(local_input_file, sep=';')

        # Initialize new columns for final decimal degrees
        df['lat_decimal'] = None
        df['lon_decimal'] = None

        # Track missing columns and skipped steps
        missing_steps = []

        # Step 1: Process existing decimal degrees
        if {'decimlat', 'decimlong'}.issubset(df.columns):
            mask_dd = pd.notna(df['decimlat']) & pd.notna(df['decimlong'])
            df.loc[mask_dd, 'lat_decimal'] = df.loc[mask_dd, 'decimlat']
            df.loc[mask_dd, 'lon_decimal'] = df.loc[mask_dd, 'decimlong']
        else:
            missing_steps.append("Step 1: Decimal Degrees (decimlat, decimlong)")

        # Step 2: Process UTM coordinates where decimal degrees are not yet set
        if {'utmeast', 'utmnorth', 'utmsrid'}.issubset(df.columns):
            mask_utm = (pd.notna(df['utmeast']) &
                       pd.notna(df['utmnorth']) &
                       pd.notna(df['utmsrid']) &
                       pd.isna(df['lat_decimal']))

            utm_results = df[mask_utm].apply(convert_utm_to_dd, axis=1)
            df.loc[mask_utm, ['lat_decimal', 'lon_decimal']] = utm_results
        else:
            missing_steps.append("Step 2: UTM Coordinates (utmeast, utmnorth, utmsrid)")

        # Step 3: Process DMS coordinates where decimal degrees are not yet set
        if {'deglat', 'minlat', 'seclat', 'deglong', 'minlong', 'seclong'}.issubset(df.columns):
            mask_dms = (pd.notna(df['deglat']) &
                       pd.notna(df['minlat']) &
                       pd.notna(df['seclat']) &
                       pd.notna(df['deglong']) &
                       pd.notna(df['minlong']) &
                       pd.notna(df['seclong']) &
                       pd.isna(df['lat_decimal']))

            dms_results = df[mask_dms].apply(convert_dms_to_dd, axis=1)
            df.loc[mask_dms, ['lat_decimal', 'lon_decimal']] = dms_results
        else:
            missing_steps.append(
                "Step 3: DMS Coordinates "
                "(deglat, minlat, seclat, deglong, minlong, seclong)"
            )

        # Save to CSV
        df.to_csv(local_output_file, sep=';', index=False)
        print(f"Processing complete. Results saved to {local_output_file}")

        # Print summary
        total_records = len(df)
        converted_records = df['lat_decimal'].notna().sum()
        print("\nConversion Summary:")
        print(f"Total records: {total_records}")
        print(f"Successfully converted: {converted_records}")
        print(f"Failed conversions: {total_records - converted_records}")

        # Include skipped steps in the summary
        if missing_steps:
            print("\nSkipped steps due to missing columns:")
            for step in missing_steps:
                print(f"- {step}")

    except FileNotFoundError as e:
        print(f"File not found: {str(e)}")
    except pd.errors.EmptyDataError as e:
        print(f"Empty data error: {str(e)}")
    except pyproj.exceptions.CRSError as e:
        print(f"CRS error: {str(e)}")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    # Get the current file's directory and move up one level to find the input file
    current_dir = Path(__file__).parent
    #input_file = current_dir.parent / "demoInputSpatialConversionData.csv"
    input_file = current_dir / "input_spatial_data.csv"
    output_file = current_dir / "output_converted_spatial_data.csv"
    process_spatial_data(str(input_file), str(output_file))
