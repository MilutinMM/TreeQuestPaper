# Code for the analysis of Tree-Quest Measurments collected in Laxenburg Park
This repository contains code necessary to reproduce the results published in (!!! the link will be added soon !!!): \
Manuscript title: Tree-Quest: A Citizen Science App for Collecting Single-Tree Information

The input datasets are published and can be downloaded from (!!! the zenodo link coming soon!!!)

## DBH/TH Analysis

### <ins>Script #1:</ins> [analyse_DBH_highQuality.py](analyse_DBH_highQuality.py) or [analyse_TH_highQuality.py](analyse_TH_highQuality.py)
The script analyzes the performance of Tree-Quest’s High-quality data wrt. FI nd TLS measurements.

- Input Datasets:
    - ARD-LP: `ard_gdf_2024_03_07_LaxPark.parquet`
    - Quality DBH:  `quality_{dbh/th}_LaxPark.parquet`
- Output Datasets: 
    - H-LP: `qualHigh_outFiltered_{dbh/th}_LaxPark`
- Output Figures: 
    - Figure 4 for DBH / Figure 5 for TH in the paper

### <ins>Script #2:</ins> [analyse_DBH_OtherQualityCats.py](analyse_DBH_OtherQualityCats.py) or [analyse_TH_OtherQualityCats.py](analyse_TH_OtherQualityCats.py)
The script analyzes the performance of Tree-Quest’s High-, Medium- and Low-quality datasets wrt. FI measurements.

- Input Datasets:
    - H-LP: `qualHigh_outFiltered_{dbh/th}_LaxPark.parquet`
    - Non H-LP: `not_qualHigh_{dbh/th}_LaxPark.parquet`
- Output Datasets: 
    - M-LP: `qualMedium_outFiltered_{dbh/th}_LaxPark.parquet`
    - L-LP: `qualLow_outFiltered_{dbh/th}_LaxPark.parquet`
- Output Figures: 
    - Figure 6 for DBH / Figure 7 for TH in the paper

### <ins>Script #3:</ins> `assign_DBH_UserName.py` or `assign_TH_UserName.py`
This script assigns the citizen category (Expert, Practitioner, Student) to each measurement. The code and the User Info input files for this script cannot be shared, as this data would expose personal user information, which, according to GDPR and the Geo-Quest user agreement, we are not allowed to share. Nevertheless, the anonymized output dataset (HML-LP) is shared to ensure reproducibility of the results.

- Input Datasets:
    - H-LP: `qualHigh_outFiltered_{dbh/th}_LaxPark.parquet`
    - M-LP: `qualMedium_outFiltered_{dbh/th}_LaxPark.parquet`
    - L-LP: `qualLow_outFiltered_{dbh/th}_LaxPark.parquet`
    - User Info files
- Output Datasets: 
    - HML-LP: `qHML_{DBH/TH}_LaxPark_UserGroups.parquet`

### <ins>Script #4:</ins> [analyse_DBH_UserGroups.py](analyse_DBH_UserGroups.py) or [analyse_TH_UserGroups.py](analyse_TH_UserGroups.py)
The script analyzes the performance of Tree-Quest’s Experts-, Practitioners-, and Students measurements wrt. FI measurements.

- Input Datasets:
    - HML-LP: `qHML_{DBH/TH}_LaxPark_UserGroups.parquet`
    - Non H-LP: `not_qualHigh_{dbh/th}_LaxPark.parquet`
- Output Figures: 
    - Figure 8 for DBH / Figure 9 for TH in the paper

### <ins>Script #5:</ins> [analyse_DBH_Method_highQuality.py](analyse_DBH_Method_highQuality.py)
The script analyzes the performance of Tree-Quest’s Automatic and Manual DBH methods, comparing with FI measurements.

- Input Datasets:
    - H-LP: `qualHigh_outFiltered_dbh_LaxPark.parquet`
- Output Figures: 
    - Figure 10 in the paper

### <ins>Script #6:</ins> [analyse_DBH_Stadtpark.py](analyse_DBH_Stadtpark.py) or [analyse_TH_Stadtpark.py](analyse_TH_Stadtpark.py)
The script compares the Tree-Quest, GreenLens, Working Trees, and GLOBE Observer measurements wrt. FI measurements.

- Input Datasets:
    - ARD-SP: `ard_gdf_2024_04_14_Stadtpark.parquet`
- Output Figures: 
    - Figure 11 for DBH / Figure 12 for TH in the paper

## Datasets Overview 

| **Dataset Name**  | **Test Site** | **Description**                                                                                                                  | **File Name**                                                              | 
|-------------------|---------------|----------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------|
| ARD-LP            | Lax. Park     | Analysis-ready dataset containing all Tree-Quest observations in Laxenburg Park                                                  | `ard_gdf_2024_03_07_LaxPark.parquet`                                       |
| Quality Data - LP | Lax. Park     | The quality scores (High, Medium, Low) for the Laxenburg park data for DBH and TH measurements                                   | `quality_dbh_LaxPark.parquet`  `quality_th_LaxPark.parquet`                |
| H-LP              | Lax. Park     | High-quality Tree-Quest measurements                                                                                             | `qualHigh_dbh_LaxPark.parquet`  `qualHigh_th_LaxPark.parquet`              |
| Not H-LP          | Lax. Park     | Tree-Quest measurements other than high-quality measurements                                                                     | `not_qualHigh_dbh_LaxPark.parquet`  `not_qualHigh_th_LaxPark.parquet`      |
| M-LP              | Lax. Park     | Medium-quality Tree-Quest measurements                                                                                           | `qualMedium_outFiltered_dbh_LaxPark.parquet`                               |
| L-LP              | Lax. Park     | Low-quality Tree-Quest measurements                                                                                              | `qualLow_outFiltered_dbh_LaxPark.parquet`                                  |
| HML-LP            | Lax. Park     | High-, Medium-, and Low-quality measurements with assigned Citizen groups (Experts, Practitioners, and Students)                 | `qHML_DBH_LaxPark_UserGroups.parquet` `qHML_TH_LaxPark_UserGroups.parquet` |
| ARD-SP            | Stadtpark     | Analysis-ready dataset containing Tree-Quest, GreenLens, Working Trees, and GLOBE Observer app observations in Stadtpark, Vienna | `ard_gdf_2024_04_14_Stadtpark.parquet`                                       |
