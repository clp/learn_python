

def calc_progress(input):
    if input < 11:
        progress_i = 1
    else:
        progress_i = max(10, int(round(input / 10)))
    return progress_i

