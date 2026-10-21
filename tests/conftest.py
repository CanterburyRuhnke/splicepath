import pytest
import torch

from splicepath.notions.config import RunConfig, load_experiment
from splicepath.patchwork.cohort import CohortBundle, SyntheticCohort
from splicepath.patchwork.dataset import Batch, collate


@pytest.fixture(scope="session")
def cfg() -> RunConfig:
    return load_experiment("_smoke")


@pytest.fixture(scope="session")
def bundle(cfg: RunConfig) -> CohortBundle:
    return SyntheticCohort(cfg).build()


@pytest.fixture()
def batch(bundle: CohortBundle) -> Batch:
    samples = [bundle.train[i] for i in range(len(bundle.train))]
    return collate(samples)


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(0)
