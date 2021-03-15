#!/usr/bin/env python

import resource
import sys


def parse_time(start, end):
    elapsed = end - start  # elapsed time in seconds
    if elapsed <= 1.0:
        ms = elapsed * 1000.
        time_string = f"{ms:.4f} ms"
    elif 1.0 < elapsed < 60.0:
        time_string = f"{elapsed:.2f} s"
    else:
        m, s = divmod(elapsed, 60)
        time_string = f"{m} min {s:.2f} s"
    return time_string


def memory_usage():
    if sys.platform == 'darwin':  # MacOS --> memory in bytes
        kB = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
    else:  # Linux --> memory in kilobytes
        kB = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    GB = kB // 1048576
    MB = kB // 1024
    kB = kB % 1024
    return GB, MB, kB