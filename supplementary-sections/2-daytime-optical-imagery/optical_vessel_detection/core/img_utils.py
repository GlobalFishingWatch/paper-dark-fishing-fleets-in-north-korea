from __future__ import print_function
from __future__ import division
import numpy as np
import skimage
from skimage import io as skio
from skimage import filters
from skimage import morphology
from skimage import transform
from collections import namedtuple
import gdal
import osr
import affine


SceneInfo = namedtuple("SceneInfo", ["path", "scene", "scene_id", "cr_to_lonlat",
                                     "angle_to_world", "pixels_to_m", "lonlat_to_cr"])


def _create_xform(path):
    # From https://stackoverflow.com/questions/2922532/obtain-latitude-and-longitude-from-a-geotiff-file
    gdal.UseExceptions()
    ds = gdal.Open(path)
    old_cs= osr.SpatialReference()
    old_cs.ImportFromWkt(ds.GetProjectionRef())

    # create the new coordinate system
    wgs84_wkt = """
    GEOGCS["WGS 84",
        DATUM["WGS_1984",
            SPHEROID["WGS 84",6378137,298.257223563,
                AUTHORITY["EPSG","7030"]],
            AUTHORITY["EPSG","6326"]],
        PRIMEM["Greenwich",0,
            AUTHORITY["EPSG","8901"]],
        UNIT["degree",0.01745329251994328,
            AUTHORITY["EPSG","9122"]],
        AUTHORITY["EPSG","4326"]]"""
    new_cs = osr.SpatialReference()
    new_cs.ImportFromWkt(wgs84_wkt)

    gt = ds.GetGeoTransform()
    c, a, b, f, d, e = gt
    gta = affine.Affine(a, b, c, d, e, f)

    # Apparently GDAL can segfault if stuff goes out of scope since it doesn't
    # handle garbage collection correctly. So 
    return old_cs, new_cs, gta, locals()


def lonlat2cr_for_geotif(path):
    """Create transform from (lon, lat) to (column, row) coordinates

    Parameters
    ----------
    path : str
        Path to a geotif file. 
    
    Returns
    ------_
    function(lon, lat) -> (column, row)

    """
    old_cs, new_cs, gta, local_vars = _create_xform(path)
    transform = osr.CoordinateTransformation(new_cs, old_cs)

    def composite(lon, lat):
        """xform from (lon, lat) to (c, r)"""
        if not -90 <= lat <= 90:
            raise ValueError('illegal lat value, did you switch coordinates')
        return (~gta * transform.TransformPoint(lat, lon)[:2])
    
    return composite



def cr2lonlat_for_geotif(path):
    """Create transform from (column, row) to (lon, lat) coordinates

    Parameters
    ----------
    path : str
        Path to a geotif file. 

    Returns
    -------
    function(columns, row) -> (lon, row)

    """
    old_cs, new_cs, gta, local_vars = _create_xform(path)
    transform = osr.CoordinateTransformation(old_cs, new_cs)

    def composite(c, r):
        """xform from (c, r) to (lon, lat)"""
        x, y = gta * (c, r)
        lat, lon = transform.TransformPoint(x, y)[:2]
        if not -90 <= lat <= 90:
            raise ValueError('illegal lat value, did you switch coordinates')
        return lon, lat
    
    return composite


def _find_angle_core(blank):
    r, c = blank.shape
    for i in range(r):
        if (~blank[i, :]).sum():
            break
    blank = blank[i:]
    for j in range(c):
        if (~blank[:, j]).sum():
            break
    blank = blank[:, j:]
    a = np.searchsorted(np.cumsum(~blank[:, 0]), 1, side='left')
    b = np.searchsorted(np.cumsum(~blank[0, :]), 1, side='left')
    return np.degrees(np.arctan2(b, a))

def _find_angle(blank):
    return 0.25 * (
                   _find_angle_core(blank) +
                   90 - _find_angle_core(blank[::-1]) +
                   90 - _find_angle_core(blank[:, ::-1]) +
                   _find_angle_core(blank[::-1, ::-1])
                )


def straighten_image(img, scale=1):
    """Straighten an image that has been rotated and has blank (alpha == 0) corners

    Parameters
    ----------
    img : np.array
        An array of RGB or RGBA values. 
    
    Returns
    --------
    np.array
        The straightend_image.
    function
        Function mapping points in original image (r, c) coordinate to new
        new images (r, c) coordinate.
    float
        The angle the image was rotated by.

    Examples
    --------
        >>> degrees_rotated, straightened_image = straighten_image(image)

    """
    img = img[::scale, ::scale]
    blank = (img[:, :, 3] == 0)
    degrees = _find_angle(blank)
    if degrees > 45:
        degrees = degrees - 90
    nr, nc = img.shape[:2]
    radians = np.radians(degrees)
    # Rotate image
    img = img.astype(float) / img.max()
    new_img = transform.rotate(img, degrees)
    new_blank = transform.rotate(blank, degrees, order=0, mode='constant', cval=1) > 0.5
    # Find clip boundaries
    valid = ~new_blank
    for i0 in range(nr):
        if valid[i0].sum():
            break
    for i1 in range(nr - 1, -1, -1):
        if valid[i1].sum():
            break
    for j0 in range(nc):
        if valid[:, j0].sum():
            break
    for j1 in range(nc - 1, -1, -1):
        if valid[:, j1].sum():
            break
    # Create transform
    center = np.array((scale * nc, scale * nr)) / 2. - 0.5
    trans = affine.Affine.translation
    xform = trans(*center) * affine.Affine.rotation(degrees) * trans(*-center) * trans(scale * j0, scale * i0)

    return new_img[i0:i1, j0:j1], xform, degrees

def create_straightened(src_path, tgt_path):
    """Save straightened version of geotiff at `src_path` to `tgt_path`.
    
    
    Parameters
    ----------
    src_path : str
        Path to source geotiff.
    tgt_path : str
        Path destination geotiff.

    """
    img = skio.imread(src_path)
    straight_img, xform, angle = straighten_image(img)
    save_geotiff(tgt_path, (straight_img * 255).astype('uint8'), xform)


def save_geotiff(path, raster, xform):
    """Save a raster image as a geotiff with associated transform

    parameters
    ----------
    path : str
        Path to save geotiff to.
    raster : np.array
        Raster data for geotiff
    xform 
        transofrm function as returned by straighten image.


    """
    # Import rasterio here, so that we don't need it in the cloud.
    import rasterio
    import affine
    n_lats, n_lons, depth = raster.shape

    assert raster.max() <= 255
    assert raster.min() >= 0
    
    profile = {
    'crs': 'EPSG:4326',
    'nodata': 0,
    'dtype': rasterio.uint8,
    'height': n_lats,
    'width': n_lons,
    'count': depth,
    'driver':"GTiff",
    'transform': xform}

    with rasterio.open(path, 'w', **profile) as dst:
        for i in range(depth):
            dst.write(raster[:, :, i], indexes=i+1)



def fill(img, sigma=1, erosion=2):
    """Fill in blank areas on the edges of images
    
    Applies a Gaussian kernel with width `sigma` to
    fill in the the first `sigma` width of unfilled 
    points. Then apply a `2 * sigma` width Gaussian
    to fill the next `2 * sigma` width of unfilled 
    points. Etc.
    
    Parameters
    ----------
    img: array of RGBA values
    sigma: float
        initial sigma of Gaussian used to extrapolate image
    erosion: int
        amount to erode the alpha mask to deal with noise at the edge

    Returns
    -------
    float array of RGBA values
        filled image   
    """
    img = img.copy()
    img = skimage.img_as_float(img)
    h, w, d = img.shape
    assert d == 4, "image must be RGBA"
    raw_mask = (img[:, :, 3] != 0)
    if raw_mask.sum() == 0:
        return img
    mask = morphology.binary_erosion(raw_mask, selem=morphology.disk(erosion))
    img[mask == 0] = 0
    invmask = ~mask
    while invmask.sum():
        denom = filters.gaussian(mask.astype(float), sigma=sigma)[invmask] + 1e-6
        for i in range(3):
            img[invmask, i] = filters.gaussian(img[:, :, i], sigma=sigma)[invmask] /  denom
        mask = morphology.binary_dilation(mask, selem=morphology.disk(sigma))
        invmask = ~mask
        img[invmask] = 0
        sigma *= 2

    img[:, :, 3] = 1.0
    return img


def fast_fill(img, sigma=1, erosion=2, max_fill=256):
    """Faster version of `fill` (usually)
    
    Try applying `fill` to only a `max_fill` area on the border
    of the image. If that is not enought to fully fill the border,
    fall back to default fill.  In the latter case this will be slower
    than the base fill.
    
    Parameters
    ---------
    img: array of RGBA values
    sigma: float
        initial sigma of Gaussian used to extrapolate image
    erosion: int
        amount to erode the alpha mask to deal with noise at the edge
    max_fill: int
        the width / height of the subimages we attempt to fill.

    Returns
    -------
    float array of RGBA values
        filled image  
    """
    h, w, d = img.shape
    if (img[:, :, 3] != 0).sum() == 0:
        return img
    if min(h, w) <= 2 * max_fill:
        return fill(img, sigma, erosion)
    new = img.copy()
    new[:max_fill] = fill(img[:max_fill], sigma, erosion)
    new[-max_fill:] = fill(img[-max_fill:], sigma, erosion)
    new[:, :max_fill] = fill(img[:, :max_fill], sigma, erosion)
    new[:, -max_fill:] = fill(img[:, -max_fill:], sigma, erosion)
    if (new[:, :, 3] == 0).sum() > 0:
        logging.warning('max_fill too small, falling back to slow fill')
        return fill(img, sigma, erosion)
    return new





    




    
