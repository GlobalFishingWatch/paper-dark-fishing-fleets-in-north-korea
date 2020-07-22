# Readme                                                                                                                                                                                                                           
                                                                                                                                                                                                                                   
## Calculate days of fishing                                                         
`calculate_fishing_days.ipynb`: This notebook calculates the days of fishing by likely Chinese fleet and N.Korean fleet in their respective fishing zone (in N.Korean waters and in Russian waters) for 2017-2018 and 2015-2018 respectively. It uses an approach using maximum daily detections for each half-month period to estimate the total fishing days by each fleet. 

## Work flow
* All detection numbers using various satellite sensors and the coast guard counts are available at `global-fishing-watch.paper_dark_fishing_fleets_in_north_korea.daily_number_of_detections`
* Run the Jupyter notebook, `calculate_fishing_days.ipynb`, to estimate the total days of fishing per group of vessels and the total fishing days. The break-down groups are as follows:
    * Chinese pair trawlers (for 2017-2018)
    * Chinese lighting vessels (for 2017-2018)
    * North Korean lighting vessels (for 2015-2018)