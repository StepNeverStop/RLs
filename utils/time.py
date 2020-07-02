import time


def get_time_hhmmss(start):
    end = time.time()
    m, s = divmod(end - start, 60)
    h, m = divmod(m, 60)
    time_str = "%02d:%02d:%02d" % (h, m, s)
    return time_str


if __name__ == "__main__":
    start = time.time()
    time.sleep(2)
    print(get_time_hhmmss(start))
