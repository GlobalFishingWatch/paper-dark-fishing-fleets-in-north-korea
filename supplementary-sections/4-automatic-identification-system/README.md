# Readme                                                                                                                                                                                                                           
                                                                                                                                                                                                                                   
## Automatic Identification System analysis                                                                                                                                                                                              
* `ais_vessel_detection_and_tracks.ipynb`: This notebook produces vessel detections and tracks based on Automatic Identification System (AIS) data.

## Work flow
* Three categories of vessels are detected in the study area using Global Fishing Watch's algorithm to detect fishing vessels and AIS data set (see Kroodsma et al. 2017).
    * Chinese fishing vessels: `global-fishing-watch.paper_dark_fishing_fleets_in_north_korea.ais_chinese_fishing_vessels_[YYYY]`
    * Non-Chinese fishing vessels: `global-fishing-watch.paper_dark_fishing_fleets_in_north_korea.ais_non_chinese_fishing_vessels_[YYYY]`
    * Non-fishing vessels: `global-fishing-watch.paper_dark_fishing_fleets_in_north_korea.ais_non_fishing_vessels_[YYYY]` 
* The tracks of the vessels detected above are stored at `global-fishing-watch.paper_dark_fishing_fleets_in_north_korea.ais_positions_[YYYY]`
* Run, then, the Jupyter notebook `ais_vessel_detection_and_tracks.ipynb` processes the data and produces AIS-based maps and tracks.