"""
timer = Timer(caption='training')
sleep(2)
timer.step("step 1", restart=True)
# will return > training: step 1: 2.00s

sleep(3)

timer.step("step 2", restart=False)
# will return > training: step 2: 3.00s

sleep(5)

timer.step("step 3", restart=True)
# will return > training: step 3: 8.00s

timer.print('step 4', restart=True) # sugar for print(timer.step('step 4', restart=True))

timer.restart(title) # sugar for timer.step(title, restart=True)

def _time() is used internally to provide the time based on optional flag provided to
Timer constructor:
'preformance' - time.perf_counter()
'process' - time.process_time()
'wall' - time.time()

which one should be the default?
"""
import time
from typing import Optional, Union


class Timer:
    def __init__(self, caption: Optional[str] = None, time: str = 'performance') -> None:
        self._caption = caption
        self._time_type = time
        self._start_time = self._time()

    def elapsed(self, restart: bool = False) -> float:
        now = self._time()
        elapsed = now - self._start_time
        if restart:
            self._start_time = now
        return elapsed

    def step(self, title: Optional[str] = None, restart: bool = False) -> str:
        elapsed = self.elapsed(restart)
        return self._format(title, elapsed)

    def restart(self, title: Optional[str] = None) -> str:
        return self.step(title, restart=True)

    def print(self, title: Optional[str] = None, restart: bool = False) -> None:
        print(self.step(title, restart))

    def _format(self, title: str, elapsed: float) -> str:
        if self._caption is None:
            caption = ""
        else:
            caption = f"{self._caption}: "
        if title is None:
            title = ""
        else:
            title = f"{title}: "
        return f"{caption}{title}{elapsed:.4f}s"

    def _time(self) -> Union[float, int]:
        if self._time_type == 'performance':
            return time.perf_counter()
        elif self._time_type == 'process':
            return time.process_time()
        elif self._time_type == 'wall':
            return time.time()
        else:
            raise ValueError(f"Unknown time type: '{self._time}'")


def _main():
    import random
    timer = Timer('training')
    time.sleep(random.random() * 3)
    print(timer.step('step 1', restart=True))
    time.sleep(random.random() * 3)
    timer.print('step 2', restart=False)
    sleep_time = random.random() * 3
    print(f"sleeping for {sleep_time}")
    time.sleep(sleep_time)
    timer.print('step 3', restart=True)
    time.sleep(random.random() * 3)
    timer.print('last step')


if __name__ == '__main__':
    _main()
