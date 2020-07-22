# Readme                                                                                                                                                                                                                           
                                                                                                                                                                                                                                   
## Visible Infrared Imaging Radiometer Suite (VIIRS) analysis                                                                                                                                                                                            
* `viirs_vessel_detections.ipynb`: This notebook provides codes that use the VIIRS dataset to detect likely Chinese and N.Korean lighting vessels in and around N.Korean waters. The number of detections and figures are used in the paper "Illuminating Dark Fishing Fleets in North Korea"

## Work flow
* Preprocessed VIIRS data sets are available below. The code processes raw VIIRS data is available at `viirs_vessel_detections.ipynb`
    * `global-fishing-watch.paper_dark_fishing_fleets_in_north_korea.viirs_east_asia_2017`
    * Vessel detections in Russian waters: `paper_dark_fishing_fleets_in_north_korea.viirs_rus_eez_north_2017`
    * Entry/exit log of S. Korean vessels operating in Russia: `paper_dark_fishing_fleets_in_north_korea.kor_entry_log_in_rus_2017`
    * Sensitivity analysis of Chinese and North Korean vessels
    * Comparison for fishing vs. non-fishing seasons: `paper_dark_fishing_fleets_in_north_korea.viirs_east_asia_2017`
    * Vessel detections in the study area: `paper_dark_fishing_fleets_in_north_korea.viirs_study_area_[YYYY]`
    * Vessel detections in the Russian EEZ: `paper_dark_fishing_fleets_in_north_korea.viirs_rus_eez_wo_12nm_2015_2018`
    * Weather data near Ulleung Islands: `paper_dark_fishing_fleets_in_north_korea.daily_cloud_near_ulleung_2017` 
* Process, then, the data sets above using `viirs_vessel_detections.ipynb` which produces maps and data to be combined with other satellite-based data
