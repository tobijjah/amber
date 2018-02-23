"""
utils.py

Author: Tobias Seydewitz
Date: 20.10.17
Mail: tobi.seyde@gmail.com

Description: Utility functions for the project "A case study on tree cover change from 2001 till 2013
             driven by illegal amber mining in the upper north oblast Rivne/Ukraine"
"""
import os
import rasterio
import numpy as np
import pandas as pd
from pathlib import Path
from rasterio import warp
from collections import namedtuple


RANDOM = np.random.RandomState(42)


def get_data_dir(path):
    """
    Recurse a directory and get all child directories paths as Path
    objects organized in a named tuple data structure.

    :param path: absolute path to directory to recurse
    :return: named tuple with directory names as key and a Path object as value
    """
    dir_structure = {
        os.path.split(root)[-1]: Path(root)
        for root, *_ in os.walk(path)
    }

    Directories = namedtuple('Directories', dir_structure.keys())
    return Directories(**dir_structure)


def read_raster(item):
    """
    Opens a raster file from string or Path object.

    :param item: path to raster as str or Path object
    :return: rasterio.DatasetReader
    """
    if isinstance(item, rasterio.io.DatasetReader):
        return item

    else:
        try:
            path = str(item)  # Cast pathlib.Path to string
            return rasterio.open(path, 'r')

        except:
            msg = 'Attr {}, Type {} is not a valid raster file'.format(item, type(item))
            raise ValueError(msg)


def fetch_metadata(from_path_or_reader, *args):
    """
    Fetches multiple metadata attributes from a raster image.

    :param from_path_or_reader: string, Path to raster image or a opened rasterio.DatasetReader
    :param args: metadata keys to fetch e.g. bounds, transform, crs etc.
    :return: metadata values
    """
    reader = read_raster(from_path_or_reader)

    values = []
    for f in args:
        value = reader.__getattribute__(f)

        if value is not None:
            values.append(value)

        else:
            raise ValueError('{} is not set'.format(f))

    if len(values) > 1:
        Metadata = namedtuple('Metadata', args)
        return Metadata(*values)

    return values[0]


def clip_raster(from_path_or_reader, dst_bounds):
    """
    Clips a raster image to target bounds.

    :param from_path_or_reader: string, Path to raster image or a opened rasterio.DatasetReader
    :param dst_bounds: target bounds left, bottom, right and top
    :return: clipped raster data and affine matrix
    """
    src = read_raster(from_path_or_reader)
    src_bounds = src.bounds

    if rasterio.coords.disjoint_bounds(src_bounds, dst_bounds):
        msg = 'Raster bounds {} are not covered by clipping bounds {}'.format(src_bounds, dst_bounds)
        raise ValueError(msg)

    window = src.window(*dst_bounds)
    window = window.round_lengths(op='ceil')
    transform = src.window_transform(window)
    data = src.read(window=window, out_shape=(src.count, window.height, window.width))

    src.close()
    return data, transform


def reproject_from(in_path, to_crs, out_path):
    """
    Reprojection of raster image to target coordinate reference system.

    :param in_path: absolute path to raster image as string
    :param to_crs: target coordinate reference system as pyproj dictionary
    :param out_path: image save path
    :return: save path as string
    """
    with rasterio.open(in_path, 'r') as src:
        affine, width, height = rasterio.warp.calculate_default_transform(
            src_crs=src.crs,
            dst_crs=to_crs,
            width=src.width,
            height=src.height,
            **src.bounds._asdict(),
        )

        kwargs = src.profile.copy()
        kwargs.update(
            transform=affine,
            width=width,
            height=height,
            crs=to_crs
        )

        with rasterio.open(out_path, 'w', **kwargs) as dst:
            for idx in src.indexes:
                rasterio.warp.reproject(
                    source=rasterio.band(src, idx),
                    destination=rasterio.band(dst, idx)
                )

        return out_path


def reproject_like(template, in_path, out_path: str):
    """
    Reprojection of a raster image like a template raster image.
    After, reprojected raster has same extent, bounds, resolution and crs
    like template raster.

    :param template: path to template raster as string, Path object or rasterio.DatasetReader
    :param in_path: path to raster image to reproject
    :param out_path: save path as string
    :return: save path as string
    """
    crs, transform, width, height = fetch_metadata(template, 'crs', 'transform', 'width', 'height')

    with rasterio.open(in_path, 'r') as src:
        out_kwargs = src.profile.copy()
        out_kwargs.update({
            'crs': crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(out_path, 'w', **out_kwargs) as dst:
            rasterio.warp.reproject(source=rasterio.band(src, list(range(1, src.count + 1))),
                                    destination=rasterio.band(dst, list(range(1, src.count + 1))))

    return out_path


def write(data, to_path, **kwargs):
    """
    Creates a raster image from a 2 or 3-dimensional numpy array.

    :param data: numpy.array
    :param to_path: save path as string
    :param kwargs: Please, consider rasterio documentation for a full list of valid of keyword arguments
    :return: save path as string
    """
    if len(data.shape) == 3:
        idx, height, width = data.shape  # z, y, x

    elif len(data.shape) == 2:
        idx = 1  # z
        height, width = data.shape  # y, x
        data = np.reshape(data.copy(), (idx, height, width))

    else:
        raise ValueError('Please, provide a valid dataset')

    dtype = data.dtype
    kwargs.update(
        count=idx,
        height=height,
        width=width,
        dtype=dtype
    )

    with rasterio.open(to_path, 'w', **kwargs) as dst:
        for i in range(idx):
            dst.write(data[i], i + 1)  # rasterio band index start at one, thus we increment by one

    return to_path


def l7_reflectance(img, esd, se, band, src_nodata=0):
    """
    Converts Landsat 7 TOA-Rad image data to Top of Atmosphere Reflectance (TOA-Ref).
    Source: https://landsat.usgs.gov/landsat-7-data-users-handbook-section-5

    :param img: radiance image data as numpy.array
    :param esd: numeric, estimated earth sun distance
    :param se: numeric, sun elevation in degrees
    :param band: band index e.g. blue = 1
    :param src_nodata: numeric, no data value in source image
    :return: image data converted to TOA-Ref
    """
    # mean solar exo-atmospheric irradiances constants see source
    esun = {
        1: 1970.0,
        2: 1842.0,
        3: 1547.0,
        4: 1044.0,
        5: 225.7,
        7: 82.06,
        8: 1369,
    }

    reflectance = (np.pi * img.astype(np.float32) * esd ** 2) / (esun[band] * np.sin(np.deg2rad(se)))

    if src_nodata is not None:
        reflectance[img == src_nodata] = 0.0

    # clip
    reflectance[reflectance < 0.0] = 0.0
    reflectance[reflectance > 1.0] = 1.0

    return reflectance


def l7_radiance(img, qcmin, qcmax, rmin, rmax, src_nodata=0):
    """
    Converts Landsat 7 quantized pixel (DN) image data to Top of Atmosphere Reflectance (TOA-Rad).
    Source: https://landsat.usgs.gov/landsat-7-data-users-handbook-section-5

    :param img: quantized pixel image data as numpy.array
    :param qcmin: numeric, Minimum quantized calibrated pixel value
    :param qcmax: numeric, Maximum quantized calibrated pixel value
    :param rmin: numeric, Minimum spectral radiance
    :param rmax: numeric, Maximum spectral radiance
    :param src_nodata: numeric, no data value in source image
    :return: image data converted to TOA-Rad
    """
    radiance = ((rmax - rmin) / (qcmax - qcmin)) * (img.astype(np.float32) - qcmin) + rmin

    if src_nodata is not None:
        radiance[img == src_nodata] = 0.0

    return radiance


def ltk_cloud_masking(red, blue, nir):
    """
    Creates a cloud mask for a Landsat 7 scene with the Luo–Trishchenko–Khlopenkov
    algorithm.

    Source: Oreopoulos et al., "Implementation on Landsat Data of a Simple
    Cloud-Mask Algorithm Developed for MODIS Land Bands", IEEE Geoscience
    and Remote Sensing Letters, vol. 8, number 4, 2011

    :param red: numpy.array, TOA-Ref image data of red band
    :param blue: numpy.array, TOA-Ref image data of blue band
    :param nir: numpy.array, TOA-Ref image data of near infrared band
    :return: cloud mask as numpy.array
    """
    if len({red.shape, blue.shape, nir.shape}) > 1:
        raise ValueError

    part1 = np.logical_or(red > 0.15, blue > 0.18)
    part2 = nir > 0.12
    part3 = np.maximum(red, blue) > nir*0.67

    mask = np.logical_and(part1, part2, part3)

    cloud_mask = np.zeros(red.shape, dtype=np.uint8)
    cloud_mask[mask] = 1

    return cloud_mask


def ndvi(red, nir):
    """
    Computes the normalized difference vegetation index for a
    Landsat scene.

    :param red: numpy.array, red band of a landsat scene
    :param nir: numpy.array, nir band of a landsat scene
    :return: ndvi as numpy.array
    """
    if red.shape != nir.shape:
        raise AttributeError('No equal shape')

    # vegetation index components
    x = nir - red
    y = nir + red

    # avoid division by zero error
    y[y == 0.0] = 1.0

    # NDVI (NIR - RED) / (NIR + RED)
    img = x / y

    img[y == 1.0] = 0.0

    # clip range to normal difference
    img[img < -1.0] = -1.0
    img[img > 1.0] = 1.0

    return img


def ndbi(swir, red, nir, blue):
    """
    Converts the swir, red, nir and blue band of Landsat scene to a
    normalized difference bareness index.

    Source: Zao and Chen, "Use of Normalized Difference Bareness Index
    in Quickly Mapping Bare from TM/ETM+", IEEE Conference Paper, 2005
    DOI: 10.1109/IGARSS.2005.1526319

    :param swir: numpy.array, shortwave infrared band
    :param red: numpy.array, red band
    :param nir: numpy.array, near infrared band
    :param blue: numpy.array, blue band
    :return: normal difference bareness index
    """
    # bareness index components
    x = (swir + red) - (nir + blue)
    y = (swir + red) + (nir + blue)

    # prevent division by zero error
    y[y == 0.0] = 1.0

    # bareness index
    img = x / y

    img[y == 1.0] = 0.0

    # clip range to normal difference
    img[img < -1.0] = -1.0
    img[img > 1.0] = 1.0

    return img


def draw_raster_sample(data, samples=100, affine=None, columns=None):
    """
    Draws a random number of samples from a 2- or 3 dimensional numpy.array and
    returns them as a pandas.DataFrame. If a affine matrix is provided the DataFrame
    contains the real world coordinates of the samples.

    :param data: numpy.array, pixel data from a raster image
    :param samples: int, number of samples to draw
    :param affine: affine matrix
    :param columns: list of string, naming of sample columns length must be equal to number of bands
    :return: raster samples as pandas.DataFrame
    """
    additional = []

    if len(data.shape) == 3:
        shapes = set([item.shape for item in data])

        if len(shapes) > 1:
            raise ValueError

        rows = RANDOM.randint(0, data[0].shape[0], samples)
        cols = RANDOM.randint(0, data[0].shape[1], samples)

    elif len(data.shape) == 2:
        rows = RANDOM.randint(0, data.shape[0], samples)
        cols = RANDOM.randint(0, data.shape[1], samples)

        data = [data]

    else:
        raise ValueError

    if affine:
        x, y = list(zip(*[cr * affine for cr in zip(cols, rows)]))
        additional.append(('x', x))
        additional.append(('y', y))

    if columns:
        if len(columns) != len(data):
            raise ValueError

        gen = ((key, data[idx]) for idx, key in enumerate(columns))

    else:
        gen = ((idx + 1, val) for idx, val in enumerate(data))

    additional.append(('row', rows))
    additional.append(('col', cols))

    samples = [(key, val[rows, cols]) for key, val in gen]

    return pd.DataFrame.from_items(samples + additional)


def confusion_matrix(label, prediction):
    """
    Creates a confusion matrix for a binary classification.

    :param label: list of int, binary class labels
    :param prediction: list of int, predicted class frequency for binary labels
    :return: 2D list respectively confusion matrix
    """
    matrix = [[0, 0],
              [0, 0]]

    for l, p in zip(label, prediction):
        if l == p:
            matrix[l][p] += 1
        else:
            matrix[l][p] += 1

    matrix[0][0], matrix[1][1] = matrix[1][1], matrix[0][0]
    return matrix
