# Readme

## Synthetic Aperture Radar analysis
* `s1_sar_vessel_detection.ipynb`: This notebook produces vessel detections on images from Sentinel-1 radar images available on Google Earth Engine. The data produced is used by the next notebook `sar_pair_trawler_detection.ipynb`.
* `sar_pair_trawler_detection.ipynb`: This notebook provides the methodology to filter pair trawlers from detected vessels using various SAR sensors including Sentinel-1, PALSAR-2, and Radarsat-2. It also produces figures used in the paper "Illuminating Dark Fishing Fleet in North Korea". 

## Work flow
* Ships are detected on various SAR images using a variation of the Constant False Alarm Rate algorithm. For Sentinel-1, use `s1_sar_vessel_detection.ipynb`. Ships detected on images of PALSAR-2 and RADARSAT-2 were provided by Japan Fisheries Research and Education Agency (FRA) and Kongsberg Satellite Services (KSAT) respectively. 
* Data on ship detections are available at
    * RADARSAT-2: `global-fishing-watch.paper_dark_fishing_fleets_in_north_korea.sar_radarsat2_detections`
    * PALSAR-22: `global-fishing-watch.paper_dark_fishing_fleets_in_north_korea.sar_palsar2_detections_[YYYYMMDD]`
    * Sentinel-1: `global-fishing-watch.paper_dark_fishing_fleets_in_north_korea.sar_sentinel1_detections`
* Once ship are detected on SAR images, we used the same model based on the distance to the nearest vessel to count pair trawlers out of the detected vessels. The codes are provided in `sar_pair_trawler_detection.ipynb`.
* The number of detected pair trawlers is available at `global-fishing-watch.paper_dark_fishing_fleets_in_north_korea.daily_number_of_detections`