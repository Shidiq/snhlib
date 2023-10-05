import re
import sys


class ProgressBar(object):
    """
    A simple progress bar for terminal
    Simple example::
        progress = ProgressBar(20, width=25, fmt=ProgressBar.FULL)
        for i in range(20):
            progress()
            progress.current = i
            sleep(0.05)
        progress.done()
        # ==> [=====                    ]  4/20 ( 20%) 16 to go
    """

    DEFAULT = "Progress: %(bar)s %(percent)3d%%"
    FULL = "%(bar)s %(current)d/%(total)d (%(percent)3d%%) %(remaining)d to go"

    def __init__(self, total, width=40, fmt=DEFAULT, symbol="=", output=sys.stderr):
        assert len(symbol) == 1

        self.total = total
        self.width = width
        self.symbol = symbol
        self.output = output
        self.fmt = re.sub(r"(?P<name>%\(.+?\))d", r"\g<name>%dd" % len(str(total)), fmt)

        self.current = 0

    def __call__(self):
        percent = self.current / float(self.total)
        size = int(self.width * percent)
        remaining = self.total - self.current
        bar = "[" + self.symbol * size + " " * (self.width - size) + "]"

        args = {
            "total": self.total,
            "bar": bar,
            "current": self.current,
            "percent": percent * 100,
            "remaining": remaining,
        }
        print("\r" + self.fmt % args, file=self.output, end="")

    def done(self):
        self.current = self.total
        self()
        print("", file=self.output)
