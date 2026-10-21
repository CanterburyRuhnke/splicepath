from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def get_logger(name: str = "splicepath") -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s %(levelname)s %(name)s :: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        root = logging.getLogger("splicepath")
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        root.propagate = False
        _CONFIGURED = True
    return logging.getLogger(name)
