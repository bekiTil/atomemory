"""Optional framework integrations.

Each submodule imports its framework at module load, so it's only pulled in when
you import it (and install the matching extra, e.g. `atomir[langchain]`). The
core package never depends on any framework.
"""
