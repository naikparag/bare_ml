import logging

V_SILENT    = 0
V_PROGRESS  = 1
V_DETAIL    = 2

def log(verbose, verbose_level, *logs):
    if verbose_level <= verbose:
        print(*logs)

def debug(*logs):
    print(*logs)