import re
import socket
import ssl

from dataclasses import dataclass

from raven.services.probes import ProbeDatabase


@dataclass
class BannerInfo:
    banner: str | None
    service: str | None
    version: str | None


class BannerGrabber:
    def __init__(self, timeout: float = 2.0, buffer_size: int = 4096):
        self.timeout = timeout
        self.buffer_size = buffer_size
        self.probes = ProbeDatabase()

    def grab(self, target: str, port: int) -> BannerInfo:
        probes = self.probes.find(port)

        if not probes:
            probes = [self._generic_probe()]

        for probe in probes:
            result = self._execute(target, port, probe)

            if result.banner and result.version:
                return result

        return self._empty_result()

    def _execute(self, target: str, port: int, probe: dict) -> BannerInfo:
        transport = probe.get("transport", "tcp").lower()

        if transport == "udp":
            return self._execute_udp(target, port, probe)

        return self._execute_tcp(target, port, probe)

    def _execute_tcp(self, target: str, port: int, probe: dict) -> BannerInfo:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        sock.settimeout(self.timeout)

        try:
            sock.connect((target, port))

            if probe.get("tls", False):
                sock = self._wrap_tls(sock, target)

            data = bytearray()

            if probe.get("wait_for_banner", False):
                data.extend(self._recv(sock))

            payload = self._build_payload(probe, target)

            if payload:
                sock.sendall(payload)

                data.extend(self._recv(sock))

            if not data:
                data.extend(self._recv(sock))

            return self._identify(self._clean(bytes(data)), probe)

        except (socket.timeout, ConnectionError, OSError, ssl.SSLError):
            return self._empty_result()

        finally:
            sock.close()

    def _execute_udp(self, target: str, port: int, probe: dict) -> BannerInfo:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        sock.settimeout(self.timeout)

        try:
            payload = self._build_payload(probe, target)

            if not payload:
                return self._empty_result()

            sock.sendto(payload, (target, port))

            data, _ = sock.recvfrom(self.buffer_size)

            return self._identify(self._clean(data), probe)

        except (socket.timeout, ConnectionError, OSError):
            return self._empty_result()

        finally:
            sock.close()

    def _recv(self, sock: socket.socket) -> bytes:
        data = bytearray()

        while len(data) < self.buffer_size:

            try:
                chunk = sock.recv(min(1024, self.buffer_size - len(data)))

                if not chunk:
                    break

                data.extend(chunk)

            except socket.timeout:
                break

            except (ConnectionError, OSError):
                break

        return bytes(data)

    @staticmethod
    def _wrap_tls(sock: socket.socket, target: str) -> ssl.SSLSocket:
        context = ssl.create_default_context()

        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE

        return context.wrap_socket(sock, server_hostname=target)

    @staticmethod
    def _build_payload(probe: dict, target: str) -> bytes:
        payload = probe.get("payload")

        if payload is None:
            return b""

        if isinstance(payload, dict):
            value = payload.get("data", "")

            encoding = payload.get("encoding", "text")

            if not isinstance(value, str):
                return b""

            value = value.replace("{target}", target)

            if encoding == "hex":
                try:
                    return bytes.fromhex(value)

                except ValueError:
                    return b""

            return value.encode()

        if isinstance(payload, str):
            return payload.replace("{target}", target).encode()

        return b""

    def _identify(self, banner: str, probe: dict) -> BannerInfo:
        if not banner:
            return self._empty_result()

        match_data = probe.get("match", {})

        if not self._matches(banner, match_data.get("contains", [])):
            return self._empty_result()

        service = match_data.get("service")

        version = None

        pattern = match_data.get("product_version")

        if pattern:
            try:
                match = re.search(pattern, banner, re.MULTILINE)

            except re.error:
                match = None

            if match:
                groups = match.groups()

                if groups:
                    product = groups[0]

                    detected_version = groups[1] if len(groups) > 1 else None

                    if product and detected_version:
                        version = f"{product} " f"{detected_version}"

                    elif product:
                        version = product

                    elif detected_version:
                        version = detected_version

        return BannerInfo(banner=banner, service=service, version=version)

    @staticmethod
    def _matches(banner: str, values: list) -> bool:
        if not values:
            return True

        banner = banner.lower()

        return any(
            isinstance(value, str) and value.lower() in banner for value in values
        )

    @staticmethod
    def _clean(data: bytes) -> str:
        return data.decode("utf-8", errors="replace").strip()

    @staticmethod
    def _empty_result() -> BannerInfo:
        return BannerInfo(banner=None, service=None, version=None)

    @staticmethod
    def _generic_probe() -> dict:

        return {
            "name": "generic",
            "ports": [],
            "transport": "tcp",
            "payload": None,
            "wait_for_banner": True,
            "match": {"contains": [], "service": None, "product_version": None},
        }
