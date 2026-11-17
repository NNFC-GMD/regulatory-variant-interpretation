"""Logging setup shared by every script in ``src/``.

Named ``log`` rather than ``logging`` so that ``import logging`` inside this
package is unambiguously the standard library.

Two things this buys over ``print``:

- **Levels.** A warning that used to be an ordinary line of stdout ("skipping N
  intervals", "[MISSING] ...") is now tagged and can be filtered. Real problems
  stop being invisible in a 10k-line SLURM log.
- **stderr.** Logs go to stderr, leaving stdout free for data. No step pipes a
  script's stdout today, but the old `print`-everything convention is what made
  that impossible to start doing.

The format deliberately matches what the bash steps emit with
``echo "[$(date '+%F %T')] ..."``, so a job's `.log` file reads consistently
whether a given line came from the wrapper or from Python.

Usage in a script::

    from utils import log

    logger = log.get_logger(__name__)

    def main():
        args = parse_args()
        log.setup(verbose=args.verbose)
        logger.info("...")

``setup()`` is idempotent, so a script that is imported rather than run does not
end up with duplicate handlers.
"""

from __future__ import annotations

import logging
import os
import sys

LOG_FORMAT = "[%(asctime)s] %(levelname)-7s %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

#: Set this in the environment to change the level without touching a command
#: line -- e.g. `export LOG_LEVEL=DEBUG` before `sbatch`, which propagates
#: to the job through --export=ALL.
LEVEL_ENV_VAR = "LOG_LEVEL"

_configured = False


def resolve_level(verbose: bool | None = None, level: int | str | None = None) -> int:
    """Pick a level from, in order: ``level``, ``verbose``, ``$LOG_LEVEL``, INFO."""
    if level is not None:
        if isinstance(level, str):
            resolved = logging.getLevelName(level.upper())
            return resolved if isinstance(resolved, int) else logging.INFO
        return level
    if verbose:
        return logging.DEBUG
    env = os.environ.get(LEVEL_ENV_VAR)
    if env:
        resolved = logging.getLevelName(env.upper())
        if isinstance(resolved, int):
            return resolved
    return logging.INFO


def setup(
    verbose: bool | None = None,
    level: int | str | None = None,
    stream=None,
    force: bool = False,
) -> logging.Logger:
    """Configure root logging once and return the root logger.

    Writes to stderr. Calling it again is a no-op unless ``force`` is set, so
    importing a script that calls it does not stack handlers.
    """
    global _configured
    root = logging.getLogger()
    if _configured and not force:
        return root

    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    root.addHandler(handler)
    root.setLevel(resolve_level(verbose=verbose, level=level))
    _configured = True
    return root


def get_logger(name: str | None = None) -> logging.Logger:
    """Logger for a module. Pass ``__name__``."""
    return logging.getLogger(name)


def add_logging_args(parser) -> None:
    """Add the standard ``--verbose`` / ``--quiet`` pair to an ArgumentParser."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="debug-level logging",
    )
    group.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="warnings and errors only",
    )


def setup_from_args(args) -> logging.Logger:
    """Configure logging from a parser that had ``add_logging_args`` applied."""
    if getattr(args, "quiet", False):
        return setup(level=logging.WARNING)
    return setup(verbose=getattr(args, "verbose", False))


def die(logger: logging.Logger, message: str, code: int = 1):
    """Log an error and exit. Replaces ``sys.exit(f"ERROR: ...")``."""
    logger.error(message)
    raise SystemExit(code)
