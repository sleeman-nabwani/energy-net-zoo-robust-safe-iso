"""SafeISO package initializer."""

# Optionally expose common submodules without forcing heavy optional deps at import time
try:  # train module may require OmniSafe and torch
    from . import train  # noqa: F401
except Exception:
    pass
try:  # eval module may require OmniSafe
    from . import eval  # noqa: F401
except Exception:
    pass

