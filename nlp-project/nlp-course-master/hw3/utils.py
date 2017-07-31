import collections
import time


class Timer(object):
    def __init__(self, part_name):
        self._times = collections.defaultdict(int)
        self._current = part_name
        self._last = time.time()

    def start_part(self, name):
        self._update()
        self._current = name

    def _update(self):
        self._times[self._current] += time.time() - self._last
        self._last = time.time()

    def __str__(self):
        self._update()
        total = sum(self._times.values())
        res = ''
        for name, elapsed in self._times.iteritems():
            res += '{} {}s {}\n'.format(name, elapsed, elapsed / total)
        return res

