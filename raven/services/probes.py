import json
from importlib.resources import files


class ProbeDatabase:
    def __init__(self):
        self.probes = self._load()

    def _load(self) -> list[dict]:
        resource = files("raven.data").joinpath("probes.json")

        with resource.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, dict):
            raise ValueError("Invalid probe database.")

        return data.get("probes", [])

    def find(self, port: int) -> list[dict]:
        return [probe for probe in self.probes if port in probe.get("ports", [])]
