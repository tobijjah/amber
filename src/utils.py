"""
utils.py

Author: Tobias Seydewitz
Date: 20.10.17
Mail: tobi.seyde@gmail.com

Description:
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
    # TODO refactor, doc
    """"""
    dir_structure = {
        os.path.split(root)[-1]: Path(root)
        for root, *_ in os.walk(path)
    }

    Directories = namedtuple('Directories', dir_structure.keys())
    return Directories(**dir_structure)


def read_raster(item):
    # TODO refactor, doc
    """"""
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
    # TODO refactor, doc
    """"""
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


def clip_raster(raster, dst_bounds):
    # TODO refactor, doc
    src = read_raster(raster)
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
    # TODO refactor, doc
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


def write(data, to_path, **kwargs):
    # TODO refactor, doc
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


def l8_reflectance(img, MR, AR, SE, src_nodata=0):
    # TODO refactor, doc
    """
    :param img: numpy.ndarray, source image
    :param MR: float, Band-specific multiplicative rescaling
    :param AR: float, Band-specific additive rescaling factor
    :param SE: float, Local sun elevation angle in degree
    :param src_nodata: int, optional no data value in source image
    :return: numpy.ndarray, image converted to TOA-Reflectance
    """
    rf = ((MR * img.astype(np.float32)) + AR) / np.sin(SE)  # conversion to rad np.deg2rad not needed

    if src_nodata is not None:
        rf[img == src_nodata] = 0.0

    return rf


def l7_reflectance(img, ESD, SE, BAND, src_nodata=0):
    # TODO refactor, doc
    """
    :param img:
    :param ESD:
    :param SE:
    :param BAND:
    :param src_nodata:
    :return:
    """
    ESUN = {
        1: 1970.0,
        2: 1842.0,
        3: 1547.0,
        4: 1044.0,
        5: 225.7,
        7: 82.06,
        8: 1369,
    }

    rf = (np.pi * img.astype(np.float32) * ESD**2) / (ESUN[BAND] * np.sin(SE))

    if src_nodata is not None:
        rf[img == src_nodata] = 0.0

    # clip
    rf[rf < 0.0] = 0.0
    rf[rf > 1.0] = 1.0

    return rf


def l7_radiance(img, QCMIN, QCMAX, RMIN, RMAX, src_nodata=0):
    # TODO doc, refactor
    """
    :param img:
    :param QCMIN:
    :param QCMAX:
    :param RMIN:
    :param RMAX:
    :param src_nodata:
    :return:
    """
    rd = ((RMAX - RMIN) / (QCMAX - QCMIN)) * (img.astype(np.float32) - QCMIN) + RMIN

    if src_nodata is not None:
        rd[img == src_nodata] = 0.0

    return rd


def ndvi(RED, NIR):
    # TODO refactor, doc, division by zero prevention
    if RED.shape != NIR.shape:
        raise AttributeError('No equal shape')

    x = NIR - RED
    y = NIR + RED

    # avoid division by zero error
    y[y == 0.0] = 1.0

    # NDVI (NIR - RED) / (NIR + RED)
    nd = x / y

    nd[y == 1.0] = 0.0

    # clip to NDVI value range
    nd[nd < -1.0] = -1.0
    nd[nd > 1.0] = 1.0

    return nd


def bi(SWIR, RED, NIR, BLUE):
    # TODO doc division by zero error
    """BI = (SWIR2 + RED - NIR - BLUE) / (SWIR2 + RED + NIR + BLUE)"""
    return (SWIR + RED - NIR - BLUE) / (SWIR + RED + NIR + BLUE)


def ndsi(SWIR, GREEN):
    # TODO doc division by zero error
    """NDSI = (SWIR2 - GREEN) / (SWIR2 + GREEN)"""
    return (SWIR - GREEN) / (SWIR + GREEN)


def draw_raster_sample(data, samples=100, affine=None, columns=None):
    # TODO doc
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
    matrix = [[0, 0],
              [0, 0]]

    for l, p in zip(label, prediction):
        if l == p:
            matrix[l][p] += 1
        else:
            matrix[l][p] += 1

    matrix[0][0], matrix[1][1] = matrix[1][1], matrix[0][0]
    return matrix


if __name__ == "__main__":
    pass
