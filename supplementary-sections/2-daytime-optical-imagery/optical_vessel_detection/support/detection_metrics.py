from glob import glob
import json
import logging
import numpy as np
import os
import skimage
import skimage.io as skio
import matplotlib.pyplot as plt
from skimage import morphology
from skimage import draw

from optical_vessel_detection.core import img_utils as iutils

from optical_vessel_detection.support import notebook_utils
from optical_vessel_detection.support.notebook_utils import path2sceneid, path2datestr


def make_xform_info(path, scale):
    """Compute` `lonlat_to_cr` transform and `base_degrees`

    Parameters
    ----------
    path : str
        Path to the geotiff file of the original, non-straightened scene
    scale: int
        Scale factor to use when creating rev_xform. Larger numbers are faster
        but do not straighten the image as accurately.

    Returns
    -------
    function
        A transform that maps lon, lat to column, row
    float
        The rotation of the scene before being straightened.
    """
    rev_xform = iutils.lonlat2cr_for_geotif(path)
    base_scene = skimage.img_as_float(skio.imread(path))
    scene, straighten_xform, base_degrees = iutils.straighten_image(base_scene, scale)
    def lonlat_to_cr(lon, lat):
        return ~straighten_xform * rev_xform(lon, lat)
    return lonlat_to_cr, base_degrees


def poly_area(x, y):
    """Return the area of a polygon using the shoelace formula.

    Based on:
    https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates

    Parameters
    ----------
    x : sequence of float
    y : sequence of float

    Returns
    -------
    float
    """
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))


km_per_deg_lat = 111
"""Approximate km per degree of latitude"""

def scene_area_km2(scene_id, scene_map):
    """Compute scene area in square kilometers

    Note
    ----
    Scenes are assumed to be small enough that we can use a constant 
    correction the squishing of longitude as we go away from the equator.

    Parameters
    ----------
    scene_id : str
    scene_map : dict
        Mapping of scene_ids to dataframes describing the scene.

    Returns
    -------
    float
        Approximate area of the scene in km^2.
    """
    bounds = json.loads(scene_map[scene_id].boundary)
    lats = [x['lat'] for x in bounds]
    lat0 = np.mean(lats)
    lon_scale = np.cos(np.radians(lat0))
    lons = np.array([lon_scale * x['lon'] for x in bounds])
    return poly_area(lats, lons) * (km_per_deg_lat ** 2)




class Detection(object):
    """An object detection

    The detection can be created either from an a CNN object detection
    (see `from_detection`) or from a human annotation (see `from_annotation`).
    Most of the use of this class comes from using the class methods find
    precision and recall by comparing CNN detection and human annotation
    data sets.

    Attributes
    ----------
    L_TO_MAJOR : float
        Conversion factor vessel length to major axis of the ellipse
        used for a boundary for the detection.
    L_TO_MINOR : float
        Conversion factor vessel length to minor axis of the ellipse
        used for a boundary for the detection.
    kinds : list of str
        List of vessel types to examine.
    annotations : dict of pd.DataFrame
        Mapping of 'YYYYMMDD' to human annotations for that date. 
        *Must be overridden*
    detections :
        Mapping of 'YYYYMMDD' to CNN detections for that date. 
        *Must be overridden*
    scenes : dict of pd.DataFrame
        Mapping of scene_ids to scene info.
        *Must be overridden*
    annotation_path_templates : list of str
        List of paths to find annotated images on.
        *Must be overridden*
    
    """
    
    _xform_info = {}

    L_TO_MAJOR = 0.75
    L_TO_MINOR = 0.375
    kinds = ['pair_trawlers', 'single_trawler', 'other_boat']

    # Subclass and override these
    annotations = None
    detections = None
    scenes = None
    annotation_path_templates = None
    
    def __init__(self, c, r, length_px, angle_rad):
        """Create Detection instance.

        Parameters
        ----------
        c : float
            Column coordinate of detected vessel.
        r : float
            Row coordinate of detected vessel.
        length_px : float
            Length of detected vessel in pixels.
        angle_rad: float
            Orientation of the detected vessel in radians. No attempt is made to
            determine the direction of travel of the vessel, so this may be
            off by 180 from the true orientation.

        Note
        ----
        All of the parameters above are referenced to the straightened
        scene image.

        """
        self.c = c
        self.r = r
        self.length_px = length_px
        self.angle_rad = angle_rad
        delta = self.L_TO_MAJOR * length_px
        self.min_c = max(int(np.floor(c - delta)), 0)
        self.max_c = int(np.ceil(c + delta)) + 1
        self.min_r = max(int(np.floor(r - delta)), 0)
        self.max_r = int(np.ceil(r + delta)) + 1
        self.angle_rad = angle_rad
        
    @classmethod 
    def _get_xform_info(cls, scene_id):
        if scene_id not in cls._xform_info:
            [path] = glob('../data/paper/scene_info/{}*.tif'.format(scene_id))
            rev_xform, base_degrees = make_xform_info(path, 16)
            cls._xform_info[scene_id] = (rev_xform, base_degrees)
        return cls._xform_info[scene_id]

    @classmethod
    def lonlat_to_cr(cls, scene_id):
        """Find or create world to image transform

        Parameters
        ----------
        scene_id : str

        Returns
        -------
        function
            A transform that maps lon, lat to column, row.
        """
        return cls._get_xform_info(scene_id)[0]
       
    @classmethod
    def degree_offset(cls, scene_id):
        """Find or compute angle between world to image coordinate

        Parameters
        ----------
        scene_id : str

        Returns
        -------
        float
            Angle between world and image coordinates
        """
        return cls._get_xform_info(scene_id)[1]

    @classmethod
    def from_annotation(cls, x):
        """Create Detection instance from human annotation
        
        Parameters
        ----------
        x : pd.Dataframe
            Row from rd.Dataframe containing annotation

        Returns
        -------
        Detection
        """
        c, r = cls.lonlat_to_cr(x.scene_id)(x.world_longitude, x.world_latitude)
        length_px = x.world_length_m / 3
        angle_rad = x.world_angle_rad
        delta_rad = np.radians(cls.degree_offset(x.scene_id))
        return cls(c, r, length_px, angle_rad + delta_rad)
    
    @classmethod
    def from_detection(cls, x):
        """Create Detection instance from CNN detection
        
        Parameters
        ----------
        x : pd.Dataframe
            Row from rd.Dataframe containing detection

        Returns
        -------
        Detection
        """
        c, r = cls.lonlat_to_cr(x.scene_id)(x.longitude, x.latitude)
        length_px = x.vessel_size / 3
        angle_rad = x.vessel_angle
        delta_rad = np.radians(cls.degree_offset(x.scene_id))
        return cls(c, r, length_px, angle_rad + delta_rad)
    
    @classmethod
    def annotation_detections_from_scene_id(cls, scene_id):
        """Convert all human annotations to Detections for a scene_id

        Parameters
        ----------
        scene_id : str

        Returns
        -------
        {str : [Detection, ...], ...}
            Dictionary has a list of Detections for each `kind` in `cls.kinds`. 
        """
        datestr = scene_id.split('_')[0]
        an_for_day = cls.annotations[datestr]
        annots = an_for_day[an_for_day.scene_id == scene_id] 
        return {k : [cls.from_annotation(x) for x in annots.itertuples() if x.kind == k] for k in cls.kinds}
    
    @classmethod
    def detection_detections_from_scene_id(cls, scene_id):
        """Convert all CNN detections to Detections for a scene_id

        Parameters
        ----------
        scene_id : str

        Returns
        -------
        {str : [Detection, ...], ...}
            Dictionary has a list of Detections for each `kind` in `cls.kinds`. 
        """
        datestr = scene_id.split('_')[0]
        det_for_day = cls.detections[datestr]
        detects = det_for_day[det_for_day.scene_id == scene_id]
        return {k : [cls.from_detection(x) for x in detects.itertuples() if x.kind == k] for k in cls.kinds}
    
    @classmethod
    def match_detections_for_scene_id(cls, scene_id):
        """Find where annotations and detections agree and disagree

        Parameters
        ----------
        scene_id : str

        Returns
        -------
        dict
            Maps `kind` of the detection to a 4-tuple of values:

            - an2det : map of annotation matched detections (true positives)
            - unmatchedan : set of unmatched annotations (false negatives)
            - det2an : map of detections to annotations (true positives)
            - unmatcheddet : set of unmatched detections (false positives)

            `an2det` and `det2an` are equivalent. 

        Note
        ----
        The matching procedure for pair_trawlers to pair_trawlers is having > 50%
        IOU. The matching for other types is intersection over annotation area
        greater than 50%. This is because these are vastly different in size. These
        other metrics are not to be used for accuracy assessments, but rather are 
        for trying to figure out why the model got a given detection wrong.

        Only the kind 'pair_trawlers' is considered for detections. The `kind`s in 
        the output mapping are the annotation `kind`s.

        """
        [path] = sum([glob(pt.format(scene_id)) for pt in cls.annotation_path_templates], [])
        img = skio.imread(path)
        mask = np.zeros(img.shape, dtype=int)
        matched = set()
        all_annot_detects = cls.annotation_detections_from_scene_id(scene_id)
        all_detect_detects = cls.detection_detections_from_scene_id(scene_id)
        results = {}
        for kind in cls.kinds:
            annot_detects = all_annot_detects[kind]
            detect_detects = all_detect_detects['pair_trawlers']
            det2an = {}
            for i, x in enumerate(detect_detects):
                for j, y in enumerate(annot_detects):
                    if y in matched:
                        # Only match a given vessel once
                        continue
                    # Skip cases where bounding boxes indicate not intersection.
                    if x.min_c > y.max_c or x.max_c < y.min_c:
                        continue
                    if x.min_r > y.max_r or x.max_r < y.min_r:
                        continue
                    # Find bounding box containing both vessels and only process that.
                    min_c = min(x.min_c, y.min_c)
                    max_c = max(x.max_c, y.max_c)
                    min_r = min(x.min_r, y.min_r)
                    max_r = max(x.max_r, y.max_r)
                    mask[min_r:max_r, min_c:max_c].fill(0)
                    # Add y vessel to mask
                    y.add_to(mask, 1)
                    if kind != 'pair_trawlers':
                        # For boats other than pair trawlers, we use the annotated
                        # are in the metrics, so compute that now.
                        annot_area = mask[min_r:max_r, min_c:max_c].sum()
                        assert annot_area > 0
                    # Add the x-vessel to the mask. The intersection is simply the
                    # area where the mask == 2 while the union is the area where the
                    # mask is greater than 0.
                    x.add_to(mask, 1)
                    intersection = (mask[min_r:max_r, min_c:max_c] == 2).sum()
                    union = (mask[min_r:max_r, min_c:max_c] > 0).sum()
                    if kind == 'pair_trawlers':
                        # Use IOU for pair trawler  to pair trawler matches.
                        metric = intersection / (union + 1e-6) # IOU
                    else:
                        # For mismatched vessel types -- pair trawler to X,
                        # use the fraction of the (typically small) annotated
                        # vessel that overlaps with the (typically large)
                        # detected pair trawler. These are used to determine if
                        # a misclassified vessel caused the false positive.
                        metric = intersection / float(annot_area)
                    # When our metric over threshold, consider it matched.
                    if metric > 0.5:
                        det2an[x] = y
                        matched.add(x)
                        break
            # Compute the inverse mapping and verify that them mapping is one to one.
            an2det = {v : k for (k, v) in det2an.items()}
            assert len(an2det) == len(det2an), "these should match since the mapping should be one to one"
            # Find the unmatched annotations and detections.
            unmatched_an = set(annot_detects) - set(an2det)
            unmatched_det = set(detect_detects) - set(det2an)
            results[kind] = (an2det, unmatched_an, det2an, unmatched_det)
        return results
        
    @classmethod
    def metrics_for_scene_ids(cls, scene_ids):
        """Compute metrics for a list of scene ids.

        Parameters
        ----------
        scene_ids : list of str
            List of scene ids to process.

        Returns
        -------
        dict
            Contains performance metrics and related parameters.
        """
        train_scene_ids = set([os.path.basename(x).rsplit('_', 2)[0] for x in 
                              glob('../data/pair_trawler_annotations/annotations/*.tif') +
                              glob('../data/pair_trawler_suppann/annotations/*.tif')])
        train_scene_ids |= notebook_utils.BAD_SCENES
        predicted = true_positives = positives = false_positives = 0
        n_fp_single_trawler = n_fp_other_boat = 0
        all_scene_ids = scene_ids
        scene_ids = [x for x in all_scene_ids if x not in train_scene_ids]
        excluded = len(all_scene_ids) - len(scene_ids)
        if excluded:
            logging.warning('Excluding {} scenes due to being in the test set'.format(excluded))
        if not scene_ids:
            logging.warning('no scene ids present')
            return None
        n_scenes = 0
        for i, sid in enumerate(scene_ids):
            matches = cls.match_detections_for_scene_id(sid)
            if matches is None:
                logging.warning('no detections for {}'.format(sid))
                continue
            n_scenes += 1
            an2det, unmatched_an, det2an, unmatched_det = matches['pair_trawlers']
            predicted += len(det2an) + len(unmatched_det)
            true_positives += len(an2det)
            positives += len(an2det) + len(unmatched_an)
            false_positives += len(unmatched_det)
            if len(unmatched_det):
                logging.info('scene {} has {} false positives'.format(sid, len(unmatched_det)))
            precision = true_positives / float(predicted + 1e-99)
            recall = true_positives / float(positives + 1e-99)
            n_fp_single_trawler += len(matches['single_trawler'][2])
            n_fp_other_boat += len(matches['other_boat'][2])
            logging.debug('cumulative to scene #{}, P={}, R={}'.format(i, precision, recall))
        area = sum([scene_area_km2(x, cls.scenes) for x in scene_ids])
        return {'precision': precision,
                'recall': recall,
                'n_predicted': predicted, 
                'n_annotated' : positives, 
                'n_true_positives' : true_positives,
                'n_false_positives' : false_positives,
                'n_fp_single_trawler' : n_fp_single_trawler,
                'n_fp_other_boat' : n_fp_other_boat,
                'n_scenes' : n_scenes,
                'area_km2' : area
                }
        
    @classmethod
    def overlay_detections_for_path(cls, path, dilations=1):
        """Load scene at path and overlay detections

        Parameters
        ----------
        path : str
            Path to a straightened scene.
        dilations : int
            Resulting ellipses will have `2 * dilations + 1` line width.

        Returns
        -------
        HxWxD np.array
            Scene indicated by path with detections overlaid on it.
        """
        img = skio.imread(path)
        matches = cls.match_detections_for_scene_id(path2sceneid(path))
        an2det, unmatched_an, det2an, unmatched_det = matches['pair_trawlers']
        img_with_annots = skimage.img_as_float(img.copy())
        rgb  = img_with_annots[:, :, :3]
        for x in an2det:
            x.draw_on(rgb, value=[0.0, 1.0, 0.0], dilations=dilations)
        for x in unmatched_det:
            x.draw_on(rgb, value=[1.0, 0.1, 0.0], dilations=dilations)
        for x in unmatched_an:
            x.draw_on(rgb, value=[1.0, 1.0, 0], dilations=dilations)
        return img_with_annots
    
    @classmethod
    def plot_detections_for_path(cls, path, dilations=1):
        """Plot the scene at path with overlaid detections

        Parameters
        ----------
        path : str
            Path to a straightened scene.
        dilations : int
            Resulting ellipses will have `2 * dilations + 1` line width.
        """
        plt.figure(figsize=(18, 6))
        plt.imshow(cls.overlay_detections_for_path(path, dilations=dilations))
        plt.show()
        
    def draw_on(self, mask, value=1, dilations=1):
        """Draw the outline of the bounding ellipse for this detection on the mask
        
        Parameters
        ----------
        mask : NxM or NxMxD np.array
        value : float or shape [D] np.array
        dilations : int
            Number of times to dilate the raw ellipse before
            setting values to zero. Resulting ellipse will have
            `2 * dilations + 1` line width.
        """
        mask_mask = np.zeros(mask.shape[:2], dtype=float)
        rr, cc = draw.ellipse_perimeter(int(round(self.r)), int(round(self.c)), 
                                        int(round(self.L_TO_MINOR * self.length_px)), int(round(self.L_TO_MAJOR*self.length_px)), 
                              orientation=-self.angle_rad, shape=mask.shape) 
        mask_mask[rr, cc] = 1
        for i in range(dilations):
            mask_mask = morphology.binary_dilation(mask_mask)
        mask[mask_mask] = value
    
    def add_to(self, mask, value=1):
        """Add value to mask to the filled bounding ellipse for this Detection

        Parameters
        ----------
        mask : HxW or HxWxD np.array
        value : float or shape [D] np.array
        """
        rr, cc = draw.ellipse(self.r, self.c, self.L_TO_MINOR * self.length_px, self.L_TO_MAJOR*self.length_px, 
                              rotation=self.angle_rad, shape=mask.shape)        
        mask[rr, cc] += value
    


