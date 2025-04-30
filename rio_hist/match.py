"""Histogram matching for raster data."""

import logging
import os

import numpy as np
import rasterio
from rasterio.transform import guard_transform

from .utils import cs_backward, cs_forward

logger = logging.getLogger(__name__)


def histogram_match(source, reference, match_proportion=1.0):
    """Adjust the values of an array so its histogram matches that of a reference array.

    Parameters
    ----------
        source: np.ndarray
        reference: np.ndarray
        match_proportion: float, range 0..1

    Returns
    -------
        target: np.ndarray
            The output array with the same shape as source
            but adjusted so that its histogram matches the reference
    """
    orig_shape = source.shape
    source = source.ravel()

    if np.ma.is_masked(reference):
        logger.debug("ref is masked, compressing")
        reference = reference.compressed()
    else:
        logger.debug("ref is unmasked, raveling")
        reference = reference.ravel()

    # get the set of unique pixel values
    # and their corresponding indices and counts
    logger.debug("Get unique pixel values")
    s_values, s_idx, s_counts = np.unique(
        source, return_inverse=True, return_counts=True
    )
    r_values, r_counts = np.unique(reference, return_counts=True)
    s_size = source.size

    if np.ma.is_masked(source):
        logger.debug("source is masked; get mask_index and remove masked values")
        mask_index = np.ma.where(s_values.mask)
        s_size = np.ma.where(s_idx != mask_index[0])[0].size
        s_values = s_values.compressed()
        s_counts = np.delete(s_counts, mask_index)

    # take the cumsum of the counts; empirical cumulative distribution
    logger.debug("calculate cumulative distribution")
    s_quantiles = np.cumsum(s_counts).astype(np.float64) / s_size
    r_quantiles = np.cumsum(r_counts).astype(np.float64) / reference.size

    # find values in the reference corresponding to the quantiles in the source
    logger.debug("interpolate values from source to reference by cdf")
    interp_r_values = np.interp(s_quantiles, r_quantiles, r_values)

    if np.ma.is_masked(source):
        logger.debug("source is masked, add fill_value back at mask_index")
        interp_r_values = np.insert(interp_r_values, mask_index[0], source.fill_value)

    # using the inverted source indicies, pull out the interpolated pixel values
    logger.debug("create target array from interpolated values by index")
    target = interp_r_values[s_idx]

    # interpolation b/t target and source
    # 1.0 = full histogram match
    # 0.0 = no change
    if match_proportion is not None and match_proportion != 1:
        diff = source - target
        target = source - (diff * match_proportion)

    if np.ma.is_masked(source):
        logger.debug("source is masked, remask those pixels by position index")
        target = np.ma.masked_where(s_idx == mask_index[0], target)
        target.fill_value = source.fill_value

    return target.reshape(orig_shape)


def calculate_mask(src, arr):
    """Calculate the mask and fill value for a raster.

    Parameters
    ----------
        src: rasterio.DatasetReader
            The source raster dataset.
        arr: np.ma.MaskedArray
            The masked array read from the source dataset.

    Returns
    -------
        mask: np.ndarray or None
            The mask array, or None if there are no masked values.
    """
    msk = arr.mask
    if msk.sum() == 0:
        mask = None
        fill = None
    else:
        _gdal_mask = src.dataset_mask()
        mask = np.invert((_gdal_mask / 255).astype("bool"))
        fill = arr.fill_value
    return mask, fill


def hist_match_worker(
    src_path,
    ref_path,
    dst_path,
    match_proportion=1.0,
    creation_options={},
    bands=[1, 2, 3],
    color_space=None,
    plot=False,
):
    """Match histogram of src to ref and write the result to dst.

    Optionally output a plot to <dst>_plot.png

    Parameters
    ----------
        src_path: str
            Path to the source raster file.
        ref_path: str
            Path to the reference raster file.
        dst_path: str
            Path to the output raster file.
        match_proportion: float
            Proportion of histogram matching (0.0 to 1.0).
        creation_options: dict
            Creation options for the output raster.
        bands: str or list of int
            Bands to be used for histogram matching.
        color_space: str, optional
            Color space to use for the histogram matching. Supported values are
            None/"RGB" (no conversion), "LCH", "LAB", "Lab", "LUV" or "XYZ".
        plot: bool
            True to create a plot of the matching process.
    """
    logger.info(
        f"Matching {os.path.basename(src_path)} to histogram of "
        f"{os.path.basename(ref_path)} using {color_space} color space"
    )

    # Validate and prepare input parameters
    if isinstance(bands, str):
        bands = [int(x) for x in bands.split(",")]
    nb_bands = len(bands)

    if color_space is not None and nb_bands != 3:
        raise ValueError("if a color_space is specified, 3 bands should be used.")

    with rasterio.open(src_path) as src_file:
        profile = src_file.profile.copy()
        src_arr = src_file.read(bands, masked=True)
        src_dtype = src_arr.dtype
        src_band_descr = src_file.descriptions
        src_mask, src_fill = calculate_mask(src_file, src_arr)
        src_arr = src_arr.filled()

    with rasterio.open(ref_path) as ref_file:
        ref_arr = ref_file.read(bands, masked=True)
        ref_mask, ref_fill = calculate_mask(ref_file, ref_arr)
        ref_arr = ref_arr.filled()

    # Some extra checks on input parameters
    if color_space is not None and src_dtype.kind == "f":
        raise ValueError("color_space must be None for floating point data.")

    # Prepare band names
    if all(src_band_descr):
        band_names = [src_band_descr[band - 1] for band in bands]
    elif color_space and len(color_space) >= max(bands) + 1:
        band_names = [color_space[x - 1] for x in bands]  # assume 1 letter per band
    else:
        band_names = [f"Band {x}" for x in bands]

    # If src file is no floating point data, normalize + apply color space conversion
    # on all bands in one go.
    if not src_dtype.kind == "f":
        src_arr = cs_forward(src_arr, color_space)
        ref_arr = cs_forward(ref_arr, color_space)

    target = src_arr.copy()

    for i, b in enumerate(bands):
        logger.debug(f"Processing band {b}")
        src_band = src_arr[i]
        ref_band = ref_arr[i]

        # For floating point data, normalize per band
        if src_dtype.kind == "f":
            max_value = max(np.max(src_band), np.max(ref_band))
            src_band = src_band / max_value
            ref_band = ref_band / max_value

        # Re-apply 2D mask to each band
        if src_mask is not None:
            logger.debug(f"apply src_mask to band {b}")
            src_band = np.ma.asarray(src_band)
            src_band.mask = src_mask
            src_band.fill_value = src_fill

        if ref_mask is not None:
            logger.debug(f"apply ref_mask to band {b}")
            ref_band = np.ma.asarray(ref_band)
            ref_band.mask = ref_mask
            ref_band.fill_value = ref_fill

        target[i] = histogram_match(src_band, ref_band, match_proportion)

        # For floating point data, denormalize per band
        if src_dtype.kind == "f":
            target[i] = (target[i] * max_value).astype(src_band.dtype)

    # No floating point data, so denormalize + convert color space for all bands.
    if not src_dtype.kind == "f":
        target = cs_backward(target, color_space)

    # re-apply src_mask to target_rgb and write ndv
    if src_mask is not None:
        logger.debug("apply src_mask to target_rgb")
        if not np.ma.is_masked(target):
            target = np.ma.asarray(target)
        target.mask = np.array((src_mask, src_mask, src_mask))
        target.fill_value = src_fill
        profile["count"] = nb_bands + 1
    else:
        profile["count"] = nb_bands

    # profile["dtype"] = "uint8"
    profile["nodata"] = None
    profile["transform"] = guard_transform(profile["transform"])
    profile.update(creation_options)

    logger.info(f"Writing raster {dst_path}")
    with rasterio.open(dst_path, "w", **profile) as dst:
        for i in range(nb_bands):
            dst.write(target[i], i + 1)
            dst.set_band_description(i + 1, band_names[i])

        if src_mask is not None:
            gdal_mask = (np.invert(src_mask) * 255).astype("uint8")
            dst.write_mask(gdal_mask)

    if plot:
        from .plot import make_plot

        outplot = os.path.splitext(dst_path)[0] + "_plot.png"
        logger.info(f"Writing figure to {outplot}")
        make_plot(
            src_path,
            ref_path,
            dst_path,
            src_arr,
            ref_arr,
            target,
            output=outplot,
            bands=tuple(zip(range(nb_bands), band_names)),
        )
