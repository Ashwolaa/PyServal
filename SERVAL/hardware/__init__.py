"""
TPX3 Hardware Interface Module

Contains data source abstractions for different acquisition hardware:
- DataSource: Abstract base class for all data sources
- SERVALSource: HTTP REST API for SERVAL detector
- Tpx3CAMSource: Direct camera access via UDP (pymepix-style)
"""

try:
    from .data_source import DataSource
    from .serval_source import SERVALSource
    from .tpx3cam_source import Tpx3CAMSource

    __all__ = [
        'DataSource',
        'SERVALSource',
        'Tpx3CAMSource',
    ]
except ImportError:
    # Source files not yet available; hardware submodule is non-functional.
    __all__ = []
