"""Fan calculator supplied by the PyMahjongGB package."""

try:
    from MahjongGB import MahjongFanCalculator
except ImportError as error:
    raise ImportError(
        "Install PyMahjongGB with pip install -r requirements.txt; see README.md"
    ) from error
