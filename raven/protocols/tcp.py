import random

from scapy.all import IP, TCP, RandShort, send, sr1

from raven.core.models import PortResult, ScanStatus


class TCPScanner:
    
    FLAGS = {
        "SYN": "S",
        "FIN": "F",
        "NULL": "",
        "XMAS": "FPU",
    }

    def __init__(self, timeout: float = 1.0):
        self.timeout = timeout

    def scan(
        self, target: str, port: int, flag: str = "SYN", ttl: int = 64
    ) -> PortResult:
        if flag not in self.FLAGS:
            raise ValueError(f"Unsupported TCP flag: {flag}")

        packet = IP(dst=target, ttl=ttl, id=random.randint(1, 65535)) / TCP(
            sport=RandShort(), dport=port, flags=self.FLAGS[flag]
        )

        try:
            response = sr1(packet, timeout=self.timeout, verbose=0)

            if response is None:
                return PortResult(port=port, protocol="tcp", status=ScanStatus.FILTERED)

            if not response.haslayer(TCP):
                return PortResult(port=port, protocol="tcp", status=ScanStatus.UNKNOWN)

            flags = int(response[TCP].flags)

            if flags & 0x12 == 0x12:
                rst = IP(dst=target) / TCP(dport=port, flags="R")

                send(rst, verbose=0)

                return PortResult(port=port, protocol="tcp", status=ScanStatus.OPEN)

            if flags & 0x14 == 0x14:
                return PortResult(port=port, protocol="tcp", status=ScanStatus.CLOSED)

            return PortResult(port=port, protocol="tcp", status=ScanStatus.FILTERED)

        except Exception as exc:
            return PortResult(
                port=port, protocol="tcp", status=ScanStatus.ERROR, error=str(exc)
            )
