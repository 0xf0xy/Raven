from dataclasses import dataclass, field
from enum import Enum


class ScanStatus(Enum):
    OPEN = "open"
    CLOSED = "closed"
    FILTERED = "filtered"
    OPEN_FILTERED = "open|filtered"
    UP = "up"
    DOWN = "down"
    UNKNOWN = "unknown"
    ERROR = "error"


@dataclass
class PortResult:
    port: int
    protocol: str
    status: ScanStatus
    banner: str | None = None
    service: str | None = None
    version: str | None = None
    error: str | None = None


@dataclass
class ScanResult:
    target: str
    ip: str
    ports: list[PortResult] = field(default_factory=list)

    @property
    def open_ports(self) -> list[PortResult]:
        return [result for result in self.ports if result.status == ScanStatus.OPEN]
