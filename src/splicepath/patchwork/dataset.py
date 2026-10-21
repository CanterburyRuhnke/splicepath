from dataclasses import dataclass
from typing import TypedDict

import torch
from torch import Tensor
from torch.utils.data import Dataset

from splicepath.templates.topology import TopologyFeatures


class Batch(TypedDict):
    embeddings: Tensor
    coords: Tensor
    psi: Tensor
    mutation: Tensor
    cnv: Tensor
    hrd: Tensor
    pseudotime: Tensor
    cancer: Tensor


@dataclass(frozen=True)
class SlideSample:
    embeddings: Tensor
    coords: Tensor
    psi: Tensor
    mutation: Tensor
    cnv: Tensor
    hrd: Tensor
    pseudotime: Tensor
    cancer: int


class SlideDataset(Dataset[SlideSample]):
    def __init__(self, samples: list[SlideSample], events: TopologyFeatures) -> None:
        self._samples = samples
        self.events = events

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> SlideSample:
        return self._samples[index]


def collate(samples: list[SlideSample]) -> Batch:
    return Batch(
        embeddings=torch.stack([s.embeddings for s in samples]),
        coords=torch.stack([s.coords for s in samples]),
        psi=torch.stack([s.psi for s in samples]),
        mutation=torch.stack([s.mutation for s in samples]),
        cnv=torch.stack([s.cnv for s in samples]),
        hrd=torch.stack([s.hrd for s in samples]),
        pseudotime=torch.stack([s.pseudotime for s in samples]),
        cancer=torch.tensor([s.cancer for s in samples], dtype=torch.long),
    )
