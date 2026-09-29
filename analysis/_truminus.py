"""Typographic minus for all figures: import this module once, after matplotlib, in every figure script.

- Tick labels built by FuncFormatter get U+2212 instead of the ASCII hyphen.
- Any other text (annotations, legends, labels) that shows a negative number gets U+2212 when the figure is saved.
Hyphens inside words or ranges (ship-hours, 2015-2023) are left alone.
"""
import re
import matplotlib.figure as _mf
import matplotlib.text as _mt
from matplotlib import ticker as _tk

_NEG = re.compile(r"(^|[\s(\[/=,:;])-(?=\.?\d)")


def true_minus(s):
    return _NEG.sub(lambda m: m.group(1) + "−", s) if isinstance(s, str) else s


_fcall = _tk.FuncFormatter.__call__
_tk.FuncFormatter.__call__ = lambda self, x, pos=None: true_minus(_fcall(self, x, pos))

_save = _mf.Figure.savefig


def _savefig(self, *a, **k):
    self.canvas.draw()
    for t in self.findobj(_mt.Text):
        s = t.get_text()
        n = true_minus(s)
        if n != s:
            t.set_text(n)
    return _save(self, *a, **k)


_mf.Figure.savefig = _savefig
