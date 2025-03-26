# Guide to Converting Spreadsheet Spatial Data to Geographic Coordinates (Decimal Degrees)

This guide outlines two methods for converting and standardizing spatial data in an Excel spreadsheet to decimal degrees (DD). The first method uses Python, while the second method utilizes PostgreSQL with Postgis extension.

These methods support spatial data in formats such as DD (Decimal Degrees), DMS (Degrees Minutes Seconds), and UTM (Universal Transverse Mercator). They can also be adapted to handle other formats.

## 1. Using Python to Convert Spreadsheet Spatial Data to Geographic Coordinates

This method is recommended for its flexibility and ease of modification to handle additional formats. It allows for automation of the conversion process and eliminates the need to remerge spatial data with the original spreadsheet.

Refer to the [SPATIAL_DATA_CONVERSION_PYTHON.MD](SPATIAL_DATA_CONVERSION_PYTHON.MD) guide for detailed instructions.

## 2. Using PostgreSQL to Convert Spreadsheet Spatial Data to Geographic Coordinates

This method explains how to use SQL and PostgreSQL to convert and standardize spatial data in an Excel spreadsheet to decimal degrees (DD).

For more details, see the [SPATIAL_DATA_CONVERSION_SQL.MD](SPATIAL_DATA_CONVERSION_SQL.MD) guide.