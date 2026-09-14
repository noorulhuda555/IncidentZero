from __future__ import annotations

from incidentzero.tools.definitions import TOOL_BY_NAME


def tools_requiring_world_version() -> frozenset[str]:
    names: set[str] = set()
    for name, spec in TOOL_BY_NAME.items():
        required = spec["function"]["parameters"].get("required", [])
        if "expected_world_version" in required:
            names.add(name)
    return frozenset(names)
