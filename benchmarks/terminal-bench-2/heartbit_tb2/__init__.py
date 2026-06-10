"""heartbit adapter for the Harbor / Terminal-Bench 2.0 harness.

Harbor resolves the agent via ``--agent-import-path heartbit_tb2.agent:HeartbitAgent``
(it imports the submodule directly). We intentionally do NOT import ``.agent``
here so that ``heartbit_tb2.heartbit_io`` (the pure helpers) can be imported and
unit-tested without Harbor installed.
"""

__all__: list[str] = []
