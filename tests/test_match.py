import numpy as np
import pytest
import rasterio

from rio_hist import match


@pytest.mark.parametrize(
    "src_path, expected_bands",
    [
        ("tests/data/source1.tif", 3),  # RGB
        ("tests/data/source2.tif", 4),  # RGBA
    ],
)
def test_hist_match_worker(tmp_path, src_path, expected_bands):
    """Test the histogram matching worker function."""
    dst_path = tmp_path / "output.tif"
    match.hist_match_worker(
        src_path=src_path,
        ref_path="tests/data/reference1.tif",
        dst_path=dst_path,
        match_proportion=0.5,
        bands=[1, 2, 3],
        color_space="RGB",
    )

    assert dst_path.exists()
    with rasterio.open(dst_path) as out:
        assert out.count == expected_bands


def test_hist_match_worker_bands(tmp_path):
    """Test the histogram matching worker function."""
    dst_path = tmp_path / "output.tif"
    match.hist_match_worker(
        src_path="tests/data/source1.tif",
        ref_path="tests/data/reference1.tif",
        dst_path=dst_path,
        match_proportion=0.5,
        bands="1,2",
    )

    assert dst_path.exists()
    with rasterio.open(dst_path) as out:
        assert out.count == 2  # RG


@pytest.mark.parametrize("color_space", ["RGB", "LCH", "LAB", "LUV", "XYZ"])
def test_hist_match_worker_colorspace(tmp_path, color_space):
    """Test the histogram matching worker function."""
    dst_path = tmp_path / "output.tif"
    match.hist_match_worker(
        src_path="tests/data/source1.tif",
        ref_path="tests/data/reference1.tif",
        dst_path=dst_path,
        match_proportion=0.5,
        bands="1,2,3",
        color_space=color_space,
    )

    assert dst_path.exists()
    with (
        rasterio.open(dst_path) as dst_file,
        rasterio.open("tests/data/source1.tif") as src_file,
        rasterio.open("tests/data/reference1.tif") as ref_file,
    ):
        assert dst_file.count == 3

        dst = dst_file.read(2)
        src = src_file.read(2)
        ref = ((ref_file.read(2) / 65365) * 256).astype("uint8")
        assert np.median(src) > np.median(dst)  # darker than the source
        assert np.median(ref) < np.median(dst)  # but not quite as dark as the reference


def test_hist_match_worker_float(tmp_path):
    """Test the histogram matching worker function with float file inputs."""
    # Convert the test file to float
    src_float_path = tmp_path / "source1_float.tif"
    with rasterio.open("tests/data/source1.tif") as src:
        data = src.read()
        meta_float = src.meta
        meta_float.update(dtype=rasterio.float32)
    with rasterio.open(src_float_path, "w", **meta_float) as ref_float:
        ref_float.write(data.astype("float32"))

    ref_float_path = tmp_path / "reference1_float.tif"
    with rasterio.open("tests/data/reference1.tif") as ref:
        data = ref.read()
        meta_float = ref.meta
        meta_float.update(dtype=rasterio.float32)
    with rasterio.open(ref_float_path, "w", **meta_float) as ref_float:
        ref_float.write(data.astype("float32"))

    # Run the test
    dst_path = tmp_path / "output.tif"
    match.hist_match_worker(
        src_path=src_float_path,
        ref_path=ref_float_path,
        dst_path=dst_path,
        match_proportion=0.5,
        creation_options={},
        bands="1,2,3",
        color_space=None,
        plot=False,
    )

    # Check the result
    assert dst_path.exists()
    with rasterio.open(dst_path) as out:
        assert out.count == 3  # RGB
