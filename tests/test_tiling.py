import numpy as np

from splicepath.patchwork.tiling import (
    macenko_stain_normalize,
    multiscale_tiles,
    otsu_threshold,
    tissue_mask,
)


def test_otsu_separates_bimodal() -> None:
    values = np.concatenate([np.full(500, 20.0), np.full(500, 200.0)])
    threshold = otsu_threshold(values)
    assert 20.0 < threshold < 200.0


def test_tissue_mask_flags_coloured_region() -> None:
    image = np.full((32, 32, 3), 240, dtype=np.uint8)
    image[:16, :16] = [150, 40, 120]
    mask = tissue_mask(image)
    assert mask[:16, :16].mean() > mask[16:, 16:].mean()


def test_multiscale_tiles_respects_sizes() -> None:
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, size=(512, 512, 3), dtype=np.uint8)
    tiles = multiscale_tiles(image, patch_sizes=(256, 512), base_size=256)
    assert set(tiles) == {256, 512}
    assert all(r >= 0 and c >= 0 for coords in tiles.values() for r, c in coords)
    assert len(tiles[512]) <= len(tiles[256])


def test_macenko_returns_same_shape() -> None:
    rng = np.random.default_rng(1)
    image = rng.integers(30, 220, size=(24, 24, 3), dtype=np.uint8)
    out = macenko_stain_normalize(image)
    assert out.shape == image.shape
    assert out.dtype == np.uint8
