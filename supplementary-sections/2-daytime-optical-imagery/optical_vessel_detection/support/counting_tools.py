import json
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib import patches, lines, path as mplpath
from mpl_toolkits.basemap import Basemap
import numpy as np

from optical_vessel_detection.support.regions import plotting_nk_eez, full_study_area


def create_valid_area_mask(lons, lats, valid_area):
    """Return mask of which detections are in the valid area.

    Parameters
    ----------
    lons : list or np.array of float
        Longitudes of detection points.
    lats : list or np.array of float
        Latitudes of detection points.
    valid_area : list of (float, float)
        Polygon defining the valid area. 
        Note first and last point should be equal.

    Returns
    -------
    np.array
        Boolean array of length `len(df)` that is True where the Detection is
        within `valid_area`
    """
    valid_poly = mplpath.Path(valid_area)
    if len(lats) == 0:
        return np.zeros([0], bool)
    pts = np.transpose([lons, lats])
    return valid_poly.contains_points(pts)


def create_aoi_mask(df, aois):
    """Return mask of which detections are inside the AOIs.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe of `Detection`s
    aois : list of [(float, float), ...[]
        List of polygon defining the AOIs. 
        Note first and last point should be equal in the polygons

    Returns
    -------
    np.array
        Boolean array of length `len(df)` that is True where the Detection is
        within one of `aois`
    """
    aoi_mask = np.zeros([len(df)], dtype=bool)
    for j, dtct in enumerate(df.itertuples()):
        if dtct.kind == 'dummy':
            continue
        for aoi in aois:
            aoi_poly = mplpath.Path(aoi)
            aoi_mask[j] |= aoi_poly.contains_point(
                            [dtct.longitude, dtct.latitude])
    return aoi_mask


def create_scene_counts(df, scene_map):
    """Return array with counts of the overlapping scenes for each detection.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe of `Detection`s
    scene_map : dict of pd.DataFrame
        Keys are YYYYMMDD strings and values are DataFrame rows containing
        scene information.

    Returns
    -------
    np.array
        Integer array of length `len(df)` that contains the count of scenes
        overlapping with each detection.
    """
    scene_counts = np.zeros(len(df), dtype=int)
    # 
    dates = set([x[:8] for x in df.scene_id])
    assert len(dates) == 1 # Not necessary, but true for what we are doing
    detect_pts = np.array([(x.longitude, x.latitude) for x in df.itertuples()])
    if len(detect_pts):
        for scene_id in scene_map:
            if scene_id[:8] in dates:
                scene = scene_map[scene_id]
                scene_bounds = json.loads(scene.boundary)
                sbound_lons = [y['lon'] for y in scene_bounds]
                sbound_lats = [y['lat'] for y in scene_bounds]
                scene_poly = mplpath.Path(np.transpose([sbound_lons, sbound_lats]))
                scene_counts += scene_poly.contains_points(detect_pts)
    return scene_counts


def find_scenes_in_aois(dates, aois, valid_area, scene_map):
    """Return scenes contained in AOIs on given dates

    Parameters
    ----------
    dates : list of str
        Dates are in `YYYYMMDD` format.
    aois : list of [(float, float), ...[]
        List of polygon defining the AOIs. 
        Note first and last point should be equal in the polygons
    valid_area : list of (float, float)
        Polygon defining the valid area. 
        Note first and last point should be equal.
    scene_map : dict of pd.DataFrame
        Keys are YYYYMMDD strings and values are DataFrame rows containing
        scene information.

    Returns
    -------
    list of str
        List of scene ids that are within *both* the valid area
        and one of the AOIs.    
    """
    scenes_in_aoi = []
    for scene_id in scene_map:
        if scene_id[:8] in dates:
            scene = scene_map[scene_id]
            scene_bounds = json.loads(scene.boundary)
            sbound_lons = [y['lon'] for y in scene_bounds]
            sbound_lats = [y['lat'] for y in scene_bounds]
            is_valid = any(create_valid_area_mask(sbound_lons, sbound_lats, valid_area))
            if is_valid:
                pts = np.transpose([sbound_lons, sbound_lats])
                for aoi in aois:
                    aoi_poly = mplpath.Path(aoi)
                    if any(aoi_poly.contains_points(pts)):
                        scenes_in_aoi.append(scene_id)
    return scenes_in_aoi


def compute_detection_counts(kinds, valid_mask, aoi_mask, scene_counts):
    """Return counts of total pair trawlers and trawler pairs.

    Parameters
    ----------
    kinds : list of str
        List of vessel kinds. Notably, 'pair_trawlers' and 'single_trawler'.
    valid_mask : np.array
        Boolean array that is true where a detection is within
        the valid region.
    aoi_mask : np.array
        Boolean array that is true where a detection is within
        on of the AOIs.
    scene_counts : np.array
        Integer array that contains a count of the number of scenes 
        overlapping each detection.

    Note
    ----
    When processing CNN detections there currently only pair trawlers
    detected, so the the first output number is simply twice the second
    one. However, when processing annotation based "detections", this is
    not true since there will also be single trawlers.
    
    Returns
    -------
    int
        The total number of detected *vessels*, the number of
        single trawlers plus twice the number of trawler pairs.
    int
        The number of detected trawler pairs.

    """
    scene_counts = np.maximum(scene_counts, 1)
    if len(kinds):
        pairs = (kinds == 'pair_trawlers')
        singles = (kinds == 'single_trawler')
        scales = (kinds == 'pair_trawlers') * 2 + (kinds == 'single_trawler')
        aoi_pts = round((scales * (valid_mask & aoi_mask) / scene_counts).sum(), 1) 
        aoi_pairs = round((pairs * (valid_mask & aoi_mask) / scene_counts).sum(), 1) 
    else:
        aoi_pts = aoi_pairs = 0
    return aoi_pts, aoi_pairs


def plot_date(date, source, scene_map, ax, aois=(), valid_area=full_study_area,
              show_scalebar=True, show_legend=True, show_xticks=True, show_yticks=True,
              show_title=True, legend_font_size=10,  font_size=10, continent_color="#b2b2b2",
              line_color='#AAAAAA',
              scene_legend_offset=(130.45, 37.83),
              detection_legend_size=6):
    """Plot detections on a given date and return the number of vessels in the AOI.

    Parameters
    ----------
    date : str
        Format is `YYYYMMDD`
    source : dict mapping str to pd.DataFrame
        Mapping of date strings to DataFrames containing all detections or
        annotations for that date.
    scene_map : dict mapping str to pd.DataFrame
        Mapping of scene ids to DataFrames containing scene information.
    ax : matplotlib.Axes
        Axes object to draw the plot on.
    aois : list of [(float, float), ...[]
        List of polygon defining the AOIs. 
        Note first and last point should be equal in the polygons
    valid_area : list of (float, float)
        Polygon defining the valid area. 
        Note first and last point should be equal.
    show_scalebar : bool
    show_legend : bool
    show_xticks : bool
    show_title : bool
    legend_font_size : int
        Size of the legend font.
    font_size : int
        Size other fonts.
    continent_color : str
    scene_legend_offset : tuple of float
        Offset of the key in the scene legend as (lon, lat).
    detection_legend_size : int
        Size of the dot used for the detection key in the legend.
    """

    if show_legend == 'minimal':
        llcrnrlon, urcrnrlon = 129, 135
        llcrnrlat, urcrnrlat = 36.9, 42.4
    else:
        llcrnrlon, urcrnrlon = 129, 135
        llcrnrlat, urcrnrlat = 38.45, 42
    projection = Basemap(llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat,
                         urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat, 
                         lat_ts=0, projection='mill', resolution="l", ax=ax)
    
    df = source[date]
    valid_mask = create_valid_area_mask(df.longitude, df.latitude, valid_area)
    aoi_mask = create_aoi_mask(df, aois)
    scene_counts = create_scene_counts(df, scene_map)
    aoi_pts, aoi_pairs = compute_detection_counts(df.kind, valid_mask, aoi_mask, scene_counts)

    # 1. plot all scenes that intersect the valid_area
    scene_count = 0
    for scene_id in scene_map:
        if scene_id.startswith(date):
            scene = scene_map[scene_id]
            bounds = json.loads(scene.boundary)
            lons = [y['lon'] for y in bounds]
            lats = [y['lat'] for y in bounds]
            is_valid = any(create_valid_area_mask(lons, lats, valid_area))
            if is_valid:
                scene_count += 1
                lons, lats = projection(lons, lats)
                ax.fill(lons, lats, color='#DDDDDD', alpha=0.25)[0]
    if show_legend:
        lons = np.array(lons)
        lats = np.array(lats)
        dx, dy = projection(scene_legend_offset[0], scene_legend_offset[1])
        lons = lons - lons.mean() + dx
        lats = lats - lats.mean() + dy
        scene_hndl = patches.Polygon(list(zip(lons, lats)), color='#DDDDDD', alpha=0.5)
        scene_hndl.set_clip_on(False)
        ax.add_patch(scene_hndl)

    # 2. Plot the NK EEZ, valid region, and the map   
    lons, lats = projection(plotting_nk_eez[:, 0], 
                            plotting_nk_eez[:, 1])
    eez_hndl = ax.plot(lons, lats, '--', color=line_color, linewidth=1)[0]
    lons, lats = projection(valid_area[:, 0], 
                            valid_area[:, 1])
    valid_hndl = ax.plot(lons, lats, '-', color=line_color, linewidth=1)[0]
    projection.fillcontinents(color=continent_color,lake_color=continent_color, zorder=2.5)

    # 3. Plot the AOIs
    aoi_hndl = None
    for aoi in aois:
        lons, lats = projection([x[0] for x in aoi], [x[1] for x in aoi])
        aoi_hndl = ax.fill(lons, lats, color='#0000FF', alpha=0.15)[0]

    # 4. Plot the detections
    lons, lats = projection(df[valid_mask].longitude.values, 
                            df[valid_mask].latitude.values)
    dtct_hndl = ax.plot(lons, lats, '.', markersize=2, 
                    color='#bd0026', label='detection', alpha=0.125)[0]
    
    # 5. Title
    title_dict = {'fontsize': font_size,}
    datestr = "{}-{}-{}".format(date[:4], date[4:6], date[6:8])
    if show_title:
        if len(aois):
            ax.set_title("{} ({} trawlers in AOI)".format(datestr, 
                         int(round(aoi_pts))), fontdict=title_dict)
        else:
            ax.set_title("{} (no fleet found)".format(datestr), 
                         fontdict=title_dict)

    # 6. Legend
    if show_legend:
        dummy = lines.Line2D([0],[0],color="w", alpha=0)
        if show_legend == 'minimal':
            title = "{} PlanetScope detections".format(datestr)
            handles = [dtct_hndl, dummy, aoi_hndl]
            labels = ['Pair trawler CNN detection',
                      "PlanetScope scene (1 of {})".format(scene_count),
                      'Location of pair trawler fleet',
                      ]
            borderpad = 0.5
            labelspacing = 0.1
            detect_index = 0
        else:
            title = None
            handles = [eez_hndl, valid_hndl, aoi_hndl, dummy, dtct_hndl]
            labels = [ 'EEZ claimed by N. Korea', 'Study area', 'Location of pair trawler fleet', 
                        "PlanetScope scene", 'Pair trawler CNN detection']
            borderpad = 0
            labelspacing = 0
            detect_index = -1

        if len(aois) == 0:
            index = labels.index("Location of pair trawler fleet")
            handles.pop(index)
            labels.pop(index)
            
        if show_legend == 'inside':
            loc = "upper right" 
            bbox_to_anchor = None
        elif show_legend == 'minimal':
            loc = 'lower center'
            bbox_to_anchor = None
        else:
            loc = "upper left"
            bbox_to_anchor = (1, 1)

        lgnd = ax.legend(handles, labels, 
                            title=title,
                            loc=loc, 
                            bbox_to_anchor=bbox_to_anchor, 
                            frameon=False,
                            fontsize=legend_font_size, 
                            borderpad=borderpad,
                            labelspacing=labelspacing,
                            framealpha=0)
        if title:
            lgnd._legend_box.sep = 5

        lgnd.legendHandles[detect_index]._legmarker.set_markersize(detection_legend_size)

        
    # 7. Scalebar
    if show_scalebar:
        font_props = {'size': font_size,}
        lon = 0.5 * (llcrnrlon + urcrnrlon)
        lat = 0.5 * (llcrnrlat + urcrnrlat)
        ([lon0, lon1], [lat0, lat1]) = projection([lon - 0.05, lon + 0.05], [lat, lat])
        scale = (lon0 - lon1) / 0.1
        scalebar = ScaleBar(111000 / scale * np.cos(np.radians(lat)), frameon=False,  
                            fixed_value=100, fixed_units='km',
                           border_pad=0.4,
                           location=1,
                           font_properties=font_props)  # 111 km / degree
        ax.add_artist(scalebar)
    
    
    # 8. Set limits and turn on/off ticks
    ax.xaxis.set_tick_params(width=0.5, color='0.5')
    ax.yaxis.set_tick_params(width=0.5, color='0.5')
    [i.set_linewidth(0.5) for i in ax.spines.values()]
    [i.set_color('0.5') for i in ax.spines.values()]
    if show_legend == 'minimal':
        raw_xticklocs = [130, 134]
        raw_yticklocs = [38, 41]
    else:
        raw_xticklocs = [130, 132, 134]
        raw_yticklocs = [39, 41]
    xticklocs, _ = projection(raw_xticklocs, raw_yticklocs[:1] * len(raw_xticklocs))
    _, yticklocs = projection(raw_xticklocs[-1:] * len(raw_yticklocs), raw_yticklocs)

    if show_xticks:
        ax.set_xticks(xticklocs)
        ax.set_xticklabels(['{}$\\degree$E'.format(x) for x in raw_xticklocs], title_dict)
    else:
        ax.set_xticks([])
    if show_yticks:
        ax.set_yticks(yticklocs)
        ax.set_yticklabels(['{}$\\degree$N'.format(x) for x in raw_yticklocs], title_dict)
    else:
        ax.set_yticks([])
        
    return aoi_pts, aoi_pairs