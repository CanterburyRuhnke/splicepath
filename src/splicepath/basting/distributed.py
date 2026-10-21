import os

import torch.distributed as dist


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_main() -> bool:
    return get_rank() == 0


def setup_ddp() -> None:
    if "RANK" in os.environ and dist.is_available() and not dist.is_initialized():
        backend = "nccl" if os.environ.get("USE_NCCL") == "1" else "gloo"
        dist.init_process_group(backend=backend)


def cleanup_ddp() -> None:
    if is_distributed():
        dist.destroy_process_group()
