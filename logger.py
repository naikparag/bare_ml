import logging

V_SILENT    = 0
V_PROGRESS  = 1
V_DETAIL    = 2

def log(verbose, verbose_level, *logs):
    if verbose_level <= verbose:
        print(*logs)

def debug(*logs):
    print(*logs)

def v_silent(verbose, *logs):
    if V_SILENT <= verbose:
        print(*logs)

def v_progress(verbose, *logs):
    if V_PROGRESS <= verbose:
        print(*logs)

def v_detail(verbose, *logs):
    if V_DETAIL <= verbose:
        print(*logs)