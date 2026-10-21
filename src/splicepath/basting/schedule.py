import math


def cosine_warmup_factor(step: int, warmup_steps: int, total_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    span = max(1, total_steps - warmup_steps)
    progress = float(step - warmup_steps) / float(span)
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))
