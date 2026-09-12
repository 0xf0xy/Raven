from scapy.all import IP, UDP, ICMP, sr1

from raven.core.models import PortResult, ScanStatus


class UDPScanner:
    def __init__(self, timeout: float = 2.0):
        self.timeout = timeout

    def scan(self, target: str, port: int, ttl: int = 64) -> PortResult:
        packet = IP(dst=target, ttl=ttl) / UDP(dport=port)

        try:
            response = sr1(packet, timeout=self.timeout, verbose=0)

            if response is None:
                return PortResult(
                    port=port, protocol="udp", status=ScanStatus.OPEN_FILTERED
                )

            if response.haslayer(UDP):
                return PortResult(port=port, protocol="udp", status=ScanStatus.OPEN)

            if response.haslayer(ICMP):
                icmp = response[ICMP]

                if icmp.type == 3 and icmp.code == 3:
                    return PortResult(
                        port=port, protocol="udp", status=ScanStatus.CLOSED
                    )

            return PortResult(port=port, protocol="udp", status=ScanStatus.UNKNOWN)

        except Exception as exc:
            return PortResult(
                port=port, protocol="udp", status=ScanStatus.ERROR, error=str(exc)
            )
