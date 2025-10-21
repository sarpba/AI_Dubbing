"""Helper utilities to keep CLI debug handling consistent."""

import logging
import warnings
from argparse import ArgumentParser

DEBUG_FLAG_NAME = "--debug"
DEBUG_FLAG_HELP = (
    "Debug mód engedélyezése: UserWarning üzenetek és DEBUG logok is megjelennek."
)


def add_debug_argument(parser: ArgumentParser) -> None:
    """Attach the shared --debug flag to an argparse parser."""
    parser.add_argument(
        DEBUG_FLAG_NAME,
        action="store_true",
        help=DEBUG_FLAG_HELP,
    )


def configure_debug_mode(enabled: bool, default_level: int = logging.INFO) -> int:
    """
    Configure how UserWarning messages behave and return the desired log level.

    Args:
        enabled: True when the caller requested debug output.
        default_level: Logging level to use when debug is disabled.

    Returns:
        The logging level that should be applied to root or named loggers.
    """
    warnings.filterwarnings(
        "default" if enabled else "ignore", category=UserWarning, append=False
    )
    return logging.DEBUG if enabled else default_level

