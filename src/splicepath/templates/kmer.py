import torch

_ALPHABET = "ACGT"
_INDEX = {base: position for position, base in enumerate(_ALPHABET)}


def kmer_dim(k: int) -> int:
    return len(_ALPHABET) ** k


def kmer_frequency(sequence: str, k: int) -> torch.Tensor:
    if k <= 0:
        raise ValueError("k must be positive")
    counts = torch.zeros(kmer_dim(k), dtype=torch.float32)
    upper = sequence.upper()
    total = 0
    for start in range(len(upper) - k + 1):
        window = upper[start : start + k]
        index = 0
        valid = True
        for base in window:
            position = _INDEX.get(base)
            if position is None:
                valid = False
                break
            index = index * len(_ALPHABET) + position
        if valid:
            counts[index] += 1.0
            total += 1
    if total > 0:
        counts /= float(total)
    return counts
