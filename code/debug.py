import time

debug_counter = 0
DEBUG = True


def debug_printer():
    global debug_counter, DEBUG
    if DEBUG:
        print("Debug: %d" % debug_counter)
        debug_counter += 1


def debug_timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        print("Run time of %s: %f sec" % (func.__name__, time.time()-start))
    return wrapper
