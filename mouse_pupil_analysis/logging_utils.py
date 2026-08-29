"""Logging setup for the command-line entry points.

Library code in this package logs and never prints, so an embedding application
controls its own output. The console scripts opt in to a plain handler here, which
keeps terminal output looking the way it always has.
"""

import logging

PACKAGE_LOGGER_NAME = "mouse_pupil_analysis"


def configure_cli_logging(level: int = logging.INFO) -> None:
    """Attach a plain stderr handler to the package logger for console use."""
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    logger.setLevel(level)
    if any(getattr(handler, "_mouse_pupil_analysis_cli", False) for handler in logger.handlers):
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler._mouse_pupil_analysis_cli = True
    logger.addHandler(handler)
    logger.propagate = False
