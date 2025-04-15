# stdlib imports --------------------------------------------------- #
from __future__ import annotations

import datetime
import logging
import logging.handlers
import pathlib
import sys
import time
import zoneinfo

# 3rd-party imports necessary for processing ----------------------- #
import dynamic_routing_analysis.codeocean_utils as co_utils

logger = logging.getLogger(__name__)


class PSTFormatter(logging.Formatter):

    def converter(self, timestamp):
        # may require 'tzdata' package
        dt = datetime.datetime.fromtimestamp(timestamp, tz=zoneinfo.ZoneInfo("UTC"))
        return dt.astimezone(zoneinfo.ZoneInfo("US/Pacific"))

    def formatTime(self, record, datefmt=None):
        dt = self.converter(record.created)
        if datefmt:
            s = dt.strftime(datefmt)
        else:
            t = dt.strftime(self.default_time_format)
            s = self.default_msec_format % (t, record.msecs)
        return s


def setup_logging(level: int | str = logging.INFO, filepath: str | None = None) -> None:
    """
    Setup logging that works for local, capsule and pipeline environments.

    - with no input arguments, log messages at INFO level and above are printed to stdout
    - in Code Ocean capsules, stdout is captured in an 'output' file automatically
    - in pipelines, stdout from each capsule instance is also captured in a central 'output' file
      - for easier reading, this function saves log files from each capsule instance individually to logs/<co_utils.AWS_BATCH_JOB_ID>.log
    - in local environments or capsules, file logging can be enabled by setting the `filepath` argument

    Note: logger is not currently safe for multiprocessing/threading (ignore WIP below)

    Note: if file logging is enabled in a multiprocessing/multithreading context, a `queue` should be set to True
    to correctly handle logs from multiple processes/threads. In this mode, a QueueListener is returned.
    When processes/threads shutdown, `QueueListener().stop()` must be called to ensure all logs are captured correctly.
    The `queue_logging()` context manager is provided to handle this within a process/thread:

        ```python
        def worker_process():
            with queue_logging():
                logger.info('Process started')
                # do work here
                logger.info('Process finished')

        processes = []
        for _ in range(5):
            process = multiprocessing.Process(target=worker_process)
            processes.append(process)
            process.start()

        for process in processes:
            process.join()
        logger.info('All processes finished')
        ```

    """
    if co_utils.is_pipeline():
        assert co_utils.AWS_BATCH_JOB_ID is not None
        co_prefix = f"{co_utils.AWS_BATCH_JOB_ID.split('-')[0]}."
    else:
        co_prefix = ""

    fmt = (
        f"%(asctime)s | %(levelname)s | {co_prefix}%(name)s.%(funcName)s | %(message)s"
    )

    formatter = PSTFormatter(  # use Seattle time
        fmt=fmt,
        datefmt="%Y-%m-%d %H:%M:%S %Z",
    )

    handlers: list[logging.Handler] = []

    # Create a console handler
    console_handler = logging.StreamHandler(stream=sys.stdout)
    handlers.append(console_handler)

    if co_utils.is_pipeline() and not filepath:
        filepath = f"/results/logs/{co_utils.AWS_BATCH_JOB_ID}_{int(time.time())}.log"
        # note: filename must be unique if we want to collect logs at end of pipeline

    if filepath:
        pathlib.Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            filename=filepath,
            maxBytes=1024 * 1024 * 10,
        )
        handlers.append(file_handler)

    # Apply formatting to the console handler and attach to root logger
    for handler in handlers:
        handler.setFormatter(formatter)
    # Configure the root logger
    logging.basicConfig(level=level, handlers=handlers)


if __name__ == "__main__":
    import doctest

    doctest.testmod(
        optionflags=(
            doctest.NORMALIZE_WHITESPACE
            | doctest.ELLIPSIS
            | doctest.IGNORE_EXCEPTION_DETAIL
        ),
    )
