from . import optimization
from . import strategy
from . import data
from . import metrics
from . import callbacks
from . import flows
from . import simulations

try:
    from importlib.metadata import version
    __version__ = version("sbisim")
except ImportError:
    # Fallback for Python < 3.8
    import pkg_resources
    __version__ = pkg_resources.get_distribution("sbisim").version

__all__ = [
    'optimization',
    'strategy',
    'data'
    'metrics', 
    'callbacks',
    'flows'
    'simulations'
]

