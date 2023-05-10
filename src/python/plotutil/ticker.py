# -*- coding: utf-8 -*-
#
# Copyright (c) Gerhard Niklasch (2023)
#
# This file is part of GWO-glitch-visualization.
#
# GWO-glitch-visualization is free software: you can redistribute it
# and/or modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation, either version 3 of
# the License, or (at your option) any later version.
#
# GWO-glitch-visualization is distributed in the hope that it will
# be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GWpy.  If not, see <http://www.gnu.org/licenses/>.

"""A custom minor formatter for use in `matplotlib` figures where the x-axis
displays frequencies between 8 Hz and just under 6000 Hz on a logarithmic
scale.
"""

# pylint: disable=E0401
from matplotlib.ticker import Formatter

# pylint: disable-next=too-few-public-methods
class MyFormatter(Formatter):
    """
    Formats (minor) tick values as integers, but avoids overcrowding
    (which also requires a judicious choice of font sizes for the major
    and minor labels).
    For use with `matplotlib.ticker.LogFormatter` (base = 10) as the
    major formatter, and adapted to the range of frequencies occurring
    in strain-based PSD or ASD plots (from 8 or 10 Hz to 8192/√2 Hz).

    (This works around a quirk in `LogFormatter`, which would happily
    produce "10^2.48" instead of "300" when a test for `x == int(x)`
    happens to fail.  It gracefully supports another `matplotlib` quirk
    viz. automatically expanding the view to a whole decade when lower
    and upper xlims coincide.)
    """
    def __init__(self, f_range):
        """
        Parameters
        ----------
        f_range : tuple
            The lower and upper (view) limit frequencies to be plotted.
        """
        self.f_low, self.f_high = f_range
        self.locs = []
        self._a_ok = False
        if self.f_high / self.f_low > 13 or self.f_high == self.f_low:
            self._good = {
                1, 2, 5, 10, 20, 50, 100, 200, 500,
                1000, 2000, 5000
            }
        elif float(self.f_high) / self.f_low > 5.6:
            self._good = {
                10, 20, 30, 40, 50, 60, 80,
                100, 200, 300, 400, 500, 600, 800,
                1000, 2000, 3000, 4000, 5000
            }
        else:
            self._good = {}
            self._a_ok = True

    # pylint: disable-next=C0103
    def __call__(self, x, pos=None):
        """
        Return the format for tick value *x* at position *pos*.
        (*pos* is passed by the caller but not used here.)
        """
        # pylint: disable-next=unused-argument
        if len(self.locs) == 0:
            return ''
        x_int = round(x)
        if abs(x - x_int) > 0.19:
            # don't label half-integers or fifth-integers in the low range
            return ''
        if self._a_ok or x_int in self._good:
            # pylint: disable-next=C0209
            return '{0:d}'.format(round(x))
        return ''
