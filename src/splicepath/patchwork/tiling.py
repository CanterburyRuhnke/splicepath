import numpy as np


def otsu_threshold(values: np.ndarray) -> float:
    flat = np.asarray(values, dtype=np.float64).ravel()
    histogram, edges = np.histogram(flat, bins=256, range=(0.0, 255.0))
    total = flat.size
    if total == 0:
        return 0.0
    centres = (edges[:-1] + edges[1:]) / 2.0
    weight = np.cumsum(histogram)
    weight_back = total - weight
    cumulative = np.cumsum(histogram * centres)
    grand = cumulative[-1]
    best_value = 0.0
    best_variance = -1.0
    for index in range(1, 256):
        wb = weight[index]
        wf = weight_back[index]
        if wb == 0 or wf == 0:
            continue
        mb = cumulative[index] / wb
        mf = (grand - cumulative[index]) / wf
        variance = wb * wf * (mb - mf) ** 2
        if variance > best_variance:
            best_variance = variance
            best_value = centres[index]
    return float(best_value)


def tissue_mask(image: np.ndarray) -> np.ndarray:
    rgb = np.asarray(image, dtype=np.float64)
    maximum = rgb.max(axis=-1)
    minimum = rgb.min(axis=-1)
    saturation = np.where(maximum > 0, (maximum - minimum) / (maximum + 1e-8), 0.0)
    scaled = saturation * 255.0
    threshold = otsu_threshold(scaled)
    return scaled > threshold


def multiscale_tiles(
    image: np.ndarray,
    patch_sizes: tuple[int, ...] = (256, 512, 1024),
    base_size: int = 256,
    tissue_fraction: float = 0.1,
) -> dict[int, list[tuple[int, int]]]:
    mask = tissue_mask(image)
    height, width = mask.shape
    result: dict[int, list[tuple[int, int]]] = {}
    for size in patch_sizes:
        coords: list[tuple[int, int]] = []
        for row in range(0, height - size + 1, base_size):
            for col in range(0, width - size + 1, base_size):
                window = mask[row : row + size, col : col + size]
                if window.mean() >= tissue_fraction:
                    coords.append((row, col))
        result[size] = coords
    return result


def macenko_stain_normalize(
    image: np.ndarray, beta: float = 0.15, alpha: float = 1.0
) -> np.ndarray:
    rgb = np.asarray(image, dtype=np.float64)
    intensity = np.clip(rgb.reshape(-1, 3), 1.0, 255.0)
    optical = -np.log(intensity / 255.0)
    tissue = optical[(optical > beta).any(axis=1)]
    if tissue.shape[0] < 3:
        return np.asarray(image, dtype=np.uint8)
    covariance = np.cov(tissue.T)
    _, vectors = np.linalg.eigh(covariance)
    plane = vectors[:, 1:3]
    projection = tissue @ plane
    angles = np.arctan2(projection[:, 1], projection[:, 0])
    low = np.percentile(angles, alpha)
    high = np.percentile(angles, 100.0 - alpha)
    first = plane @ np.array([np.cos(low), np.sin(low)])
    second = plane @ np.array([np.cos(high), np.sin(high)])
    stains = np.stack([first, second], axis=1)
    if stains[0, 0] < stains[0, 1]:
        stains = stains[:, ::-1]
    concentrations = np.linalg.lstsq(stains, optical.T, rcond=None)[0]
    maxima = np.percentile(concentrations, 99.0, axis=1, keepdims=True)
    maxima[maxima == 0.0] = 1.0
    normalized = concentrations / maxima
    reference = np.array([[1.9705, 1.0308], [1.0, 1.0], [0.0, 0.6170]])
    rebuilt = reference @ normalized
    restored = 255.0 * np.exp(-rebuilt.T)
    restored = np.clip(restored, 0.0, 255.0).reshape(rgb.shape)
    return restored.astype(np.uint8)
