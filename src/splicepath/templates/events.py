from dataclasses import dataclass
from enum import IntEnum


class EventType(IntEnum):
    ES = 0
    A5 = 1
    A3 = 2
    RI = 3
    ME = 4


EVENT_TYPES: tuple[EventType, ...] = (
    EventType.ES,
    EventType.A5,
    EventType.A3,
    EventType.RI,
    EventType.ME,
)


@dataclass(frozen=True)
class SpliceEvent:
    event_id: str
    gene: str
    event_type: EventType
    upstream_exon_len: int
    downstream_exon_len: int
    intron_len: int
    junction_sequence: str

    def __post_init__(self) -> None:
        if self.upstream_exon_len <= 0 or self.downstream_exon_len <= 0:
            raise ValueError("exon lengths must be positive")
        if self.intron_len <= 0:
            raise ValueError("intron length must be positive")
