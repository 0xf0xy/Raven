from concurrent.futures import ThreadPoolExecutor

from raven.core.models import ScanResult, PortResult
from raven.core.resolver import TargetResolver
from raven.protocols.tcp import TCPScanner
from raven.protocols.udp import UDPScanner
from raven.protocols.icmp import ICMPScanner
from raven.services.banner import BannerGrabber

from importlib.resources import files


class Raven:
    def __init__(self, timeout: float = 1.0, workers: int = 50):
        self.timeout = timeout
        self.workers = workers

        with files("raven.data").joinpath("common.txt").open("r") as file:
            self.common_ports = [int(line.strip()) for line in file if line.strip()]

        self.tcp = TCPScanner(timeout)
        self.udp = UDPScanner(timeout)
        self.icmp = ICMPScanner(timeout)
        self.banner = BannerGrabber(timeout)

    def scan_tcp(
        self, target: str, ports: list[int], flag: str = "SYN", ttl: int = 64
    ) -> ScanResult:
        ip = TargetResolver.resolve(target)

        results = []

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = [
                executor.submit(self.tcp.scan, ip, port, flag, ttl) for port in ports
            ]

            for future in futures:
                results.append(future.result())

        return ScanResult(
            target=target, ip=ip, ports=sorted(results, key=lambda result: result.port)
        )

    def scan_udp(self, target: str, ports: list[int], ttl: int = 64) -> ScanResult:
        ip = TargetResolver.resolve(target)

        results = []

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = [executor.submit(self.udp.scan, ip, port, ttl) for port in ports]

            for future in futures:
                results.append(future.result())

        return ScanResult(
            target=target, ip=ip, ports=sorted(results, key=lambda result: result.port)
        )

    def scan_icmp(self, target: str):
        ip = TargetResolver.resolve(target)
        return self.icmp.scan(ip)

    def grab_banners(self, target: str, ports: list[int]) -> ScanResult:
        ip = TargetResolver.resolve(target)

        results = []

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = [executor.submit(self.banner.grab, ip, port) for port in ports]

            for port, future in zip(ports, futures):
                info = future.result()

                results.append(
                    PortResult(
                        port=port,
                        protocol="tcp",
                        status="open",
                        banner=info.banner,
                        service=info.service,
                        version=info.version,
                    )
                )

        return ScanResult(
            target=target, ip=ip, ports=sorted(results, key=lambda result: result.port)
        )
