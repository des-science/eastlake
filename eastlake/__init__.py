try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version("eastlake")
except PackageNotFoundError:
    # package is not installed
    pass

from .pipeline import register_pipeline_step  # noqa
from . import des_piff  # noqa
from . import des_smoothpiff  # noqa
