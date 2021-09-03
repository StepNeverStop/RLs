# Added from https://github.com/danijar/dreamerv2/blob/main/dreamerv2/common/when.py

class Every:

    def __init__(self, every):
        self._every = every
        self._last = None

    def __call__(self, step):
        step = int(step)
        if not self._every:
            return False
        if self._last is None:
            self._last = step
            return True
        if step >= self._last + self._every:
            self._last += self._every
            return True
        return False


class Once:

    def __init__(self):
        self._once = True

    def __call__(self):
        if self._once:
            self._once = False
            return True
        return False


class Until:

    def __init__(self, until=None):
        self._until = until

    def __call__(self, step):
        step = int(step)
        if not self._until:
            return True
        return step < self._until


if __name__ == '__main__':

    e = Every(10)
    for i in range(100):
        if e(i):
            print(i)

    o = Once()
    if o():
        print('first')
    if o():
        print('second')

    u = Until(1)
    for i in range(10):
        if u(i):
            print(i)
