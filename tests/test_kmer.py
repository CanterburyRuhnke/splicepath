import torch

from splicepath.templates.kmer import kmer_dim, kmer_frequency


def test_kmer_dim() -> None:
    assert kmer_dim(3) == 64
    assert kmer_dim(1) == 4


def test_kmer_frequency_normalised() -> None:
    freq = kmer_frequency("ACGTACGTACGT", 3)
    assert freq.shape == (64,)
    assert torch.isclose(freq.sum(), torch.tensor(1.0))


def test_kmer_counts_known_sequence() -> None:
    freq = kmer_frequency("AAAA", 2)
    aa_index = 0
    assert freq[aa_index].item() == 1.0
    assert freq.sum().item() == 1.0


def test_kmer_ignores_invalid_bases() -> None:
    freq = kmer_frequency("ACNNGT", 2)
    assert torch.isclose(freq.sum(), torch.tensor(1.0))
