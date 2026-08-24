# fzgm_helper.py -- small convenience layer over the fzgpumodules libpressio
# plugin's early_config format. Not part of libpressio; a thin builder that
# saves hand-writing "sN <- sM:port" connection strings and
# "fzgpumodules:sN:<option>" keys for the common case of a linear pipeline.
# See docs/libpressio_python.md for the full option surface this wraps.

class Chain:
    """Builds a linear fzgpumodules stage pipeline.

    Each add() call appends one stage, auto-numbers it sN, and wires it to
    the previous stage's output (default port, or a named one via
    from_port). Keyword args become that stage's "fzgpumodules:sN:<key>"
    options.

    >>> Chain().add("lorenzo:float:uint16", quant_radius=999) \\
    ...        .add("rze", from_port="codes", chunk_size=8192) \\
    ...        .early_config()
    {'fzgpumodules:stages': ['lorenzo:float:uint16', 'rze'],
     'fzgpumodules:connections': ['s1 <- s0:codes'],
     'fzgpumodules:s0:quant_radius': 999,
     'fzgpumodules:s1:chunk_size': 8192}
    """

    def __init__(self):
        self._stages = []
        self._connections = []
        self._options = {}
        self._top = {}

    def add(self, token, from_port=None, **options):
        sid = f"s{len(self._stages)}"
        if self._stages:
            prev = f"s{len(self._stages) - 1}"
            src = f"{prev}:{from_port}" if from_port else prev
            self._connections.append(f"{sid} <- {src}")
        elif from_port is not None:
            raise ValueError("from_port has no effect on the first stage (nothing precedes it)")
        self._stages.append(token)
        for key, value in options.items():
            self._options[f"fzgpumodules:{sid}:{key}"] = value
        return self

    def configure(self, **top_level_options):
        """Set top-level fzgpumodules:* options, e.g. fusion="auto", error_bound_mode="prel"."""
        for key, value in top_level_options.items():
            self._top[f"fzgpumodules:{key}"] = value
        return self

    def early_config(self):
        cfg = {
            "fzgpumodules:stages": list(self._stages),
            "fzgpumodules:connections": list(self._connections),
        }
        cfg.update(self._top)
        cfg.update(self._options)
        return cfg

    def compressor(self, eb=None, metric="size"):
        """Build a ready-to-use PressioCompressor from the chain built so far."""
        import libpressio as lp
        compressor_config = {"pressio:metric": metric}
        if eb is not None:
            compressor_config["pressio:abs"] = eb
        return lp.PressioCompressor.from_config({
            "compressor_id": "fzgpumodules",
            "early_config": self.early_config(),
            "compressor_config": compressor_config,
        })
