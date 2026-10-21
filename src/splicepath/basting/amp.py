from contextlib import AbstractContextManager, nullcontext

import torch


def autocast_context(
    device: torch.device, enabled: bool, dtype_name: str = "bfloat16"
) -> AbstractContextManager[None]:
    if not enabled or device.type != "cuda":
        return nullcontext()
    dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)
