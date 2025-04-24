import rasterio

from rio_hist import match


def test_hist_match_worker(tmp_path):
    """Test the histogram matching worker function."""
    dst_path = tmp_path / "output.tif"
    match.hist_match_worker(
        src_path="tests/data/source1.tif",
        ref_path="tests/data/reference1.tif",
        dst_path=dst_path,
        match_proportion=0.5,
        creation_options={},
        bands="1,2,3",
        color_space="RGB",
        plot=False,
    )

    assert dst_path.exists()
    with rasterio.open(dst_path) as out:
        assert out.count == 3  # RGB


def test_hist_match_worker_bands(tmp_path):
    """Test the histogram matching worker function."""
    dst_path = tmp_path / "output.tif"
    match.hist_match_worker(
        src_path="tests/data/source1.tif",
        ref_path="tests/data/reference1.tif",
        dst_path=dst_path,
        match_proportion=0.5,
        creation_options={},
        bands="1,2",
        color_space="RGB",
        plot=False,
    )

    assert dst_path.exists()
    with rasterio.open(dst_path) as out:
        assert out.count == 2  # RG
