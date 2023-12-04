import time

class MTimer:
    def __init__(self):
        self.records = {}

    def _time():
        return time.process_time()

    def start(self, key):
        if key not in self.records:
            self.records[key] = {'total_time': 0, 'calls': 0, 'start_time': None}
        self.records[key]['start_time'] = self._time()

    def end(self, key):
        if key in self.records and self.records[key]['start_time'] is not None:
            end_time = self._time()
            elapsed = end_time - self.records[key]['start_time']
            self.records[key]['total_time'] += elapsed
            self.records[key]['calls'] += 1
            self.records[key]['start_time'] = None

    def dump(self):
        for key in self.records:
            total_time = self.records[key]['total_time']
            calls = self.records[key]['calls']
            avg_time = total_time / calls if calls > 0 else 0
            print(f'{key}: {calls} calls, {total_time:.3f} sec, {avg_time:.3f} s/c')

    def reset(self):
        self.records = {}
