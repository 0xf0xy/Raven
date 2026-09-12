import socket


class TargetResolver:
    @staticmethod
    def resolve(target: str) -> str:
        try:
            return socket.gethostbyname(target)

        except socket.gaierror as exc:
            raise ValueError(f"Could not resolve host: {target}") from exc
