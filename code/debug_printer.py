
debug_counter = 0
DEBUG = True


def debug_printer():
    global debug_counter, DEBUG
    if DEBUG:
        print("Debug: %d" % debug_counter)
        debug_counter += 1
