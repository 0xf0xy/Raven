class PortParser:
    @staticmethod
    def parse(value: str | None, common_ports: list[int]) -> list[int]:
        if not value:
            return common_ports.copy()

        ports: set[int] = set()

        for item in value.split(","):
            item = item.strip()

            if not item:
                continue

            if "-" in item:
                start, end = item.split("-", 1)

                start = int(start)
                end = int(end)

                if start > end:
                    raise ValueError(f"Invalid port range: {item}")

                ports.update(range(start, end + 1))

            else:
                ports.add(int(item))

        for port in ports:
            if not 1 <= port <= 65535:
                raise ValueError(f"Invalid port: {port}")

        return sorted(ports)
