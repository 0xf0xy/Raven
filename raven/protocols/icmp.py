from scapy.all import IP, ICMP, sr1

from raven.core.models import ScanStatus


class ICMPScanner:
    def __init__(self, timeout: float = 2.0):
        self.timeout = timeout

    def scan(self, target: str) -> ScanStatus:
        packet = IP(dst=target) / ICMP()

        try:
            response = sr1(packet, timeout=self.timeout, verbose=0)

            if response and response.haslayer(ICMP):
                if response[ICMP].type == 0:
                    return ScanStatus.UP

                if response[ICMP].type == 3:
                    return ScanStatus.UP

            return ScanStatus.DOWN

        except Exception:
            return ScanStatus.UNKNOWN
