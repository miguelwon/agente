# In your agente/__init__.py
try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"