"""Common info used by notebooks.
"""

fully_annotated = {
    '20170627',
    '20170806',
    '20171024',
    '20180602',
    '20180715',
    '20180826',
    '20181008'
}
"""Date that we fully annotated the chosen AOIs"""

def extents_to_aoi(ll, ur):
    """Convert ll-corner and ur-corner to rectangular AOI

    Parameters
    ----------
    ll : tuple of float
        (lon, lat) of lower left corner of AOI
    ur : tuple of float
        (lon, lat) of upper right corner of AOI

    Returns
    -------
    list
        5 (lon, lat) pairs defining a rectangle. The last pair is a 
        repeat of the first
    """
    (lon0, lat0) = ll
    (lon1, lat1) = ur
    return [
                (lon0, lat0),
                (lon1, lat0),
                (lon1, lat1),
                (lon0, lat1),
                (lon0, lat0),
            ]



aois_by_date = {
    '20170528' : [extents_to_aoi((130.5, 39.15), (132, 40.45))],
    '20170627' : [extents_to_aoi((131.75, 39.45), (132.65, 40.3))],
    '20170715' : [extents_to_aoi((131.1, 40.3), (132.1, 41.35))],
    '20170806' : [extents_to_aoi((131.2, 39.5), (132.4, 40.25))],
    '20170908' : [extents_to_aoi((131.0, 40.3), (132.1, 41.35)),
                  extents_to_aoi((131.7, 39.7), (132.95, 39.95))],
    '20170926' : [[
                  (131.7, 39.3),
                  (133.35, 39.3),
                  (133.85, 40.4),
                  (132.2, 40.4),
                  (131.7, 39.3)]],
    '20171024' : [[(131.4, 40.1), (132.7, 40.1), 
                   (132.7, 40.55), (132.0, 41.25),
                   (131.4, 41.25), (131.4, 40.1)]],
    '20171105' : [extents_to_aoi((132.45, 39.65), (133.45, 40.25))],

    '20180421' : [],
    '20180514' : [extents_to_aoi((129.9, 39.0), (131.15, 39.6))], # TODO: None -> no fleet detected
    '20180522' : [extents_to_aoi((131.25, 39.0), (132.85, 40.1))],
    '20180602' : [[(129.9, 38.95),
                   (132.3, 38.95),
                   (132.6, 39.9),
                   (130.2, 39.9),
                   (129.9, 38.95)]],
    '20180622' : [extents_to_aoi((131.8, 39.2), (132.6, 40.2))],
    '20180715' : [extents_to_aoi((131.7, 39.45), (132.8, 40.1))],
    '20180731' : [extents_to_aoi((132.1, 39.2), (132.8, 40.2))],
    '20180814' : [extents_to_aoi((132.33, 39.65), (133.38, 40.25))],
    '20180826' : [extents_to_aoi((132.55, 39.45), (133.42, 40.0))],
    '20180828' : [extents_to_aoi((132.55, 39.45), (133.42, 40.0))],
    '20180911' : [extents_to_aoi((132.4, 39.7), (133.2, 40.1))],
    '20180912' : [extents_to_aoi((132.2, 39.7), (133.0, 40.1))],
    '20180917' : [extents_to_aoi((132.35, 38.95), (133.15, 39.6))],
    '20180927' : [extents_to_aoi((131.6, 39.4), (132.8, 39.9))],
    '20180929' : [extents_to_aoi((132.4, 38.95), (133.2, 39.8))],
    '20181008' : [extents_to_aoi((132.4, 38.9), (133.5, 39.7))],
    '20181102' : [extents_to_aoi((133.0, 39.55), (133.84, 40.0))], 
    '20181115' : [[(130.75, 39.25),
                   (132.37, 39.25),
                   (132.63, 40.1),
                   (131.01, 40.1),
                   (130.75, 39.25)]],
}
"""Mapping of dates to lists of AOIs. These AOIs should enclose the dense parts of the fleet.


Each AOI is a list if (lon, lat) pairs with the last pair being the same as the first in order
to close the polygon. So the overall structure is [[(lon, lat), ...], ...]
"""
