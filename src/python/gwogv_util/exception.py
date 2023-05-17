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
# along with GWO-glitch-visualization.
# If not, see <http://www.gnu.org/licenses/>.

"""Custom exception classes"""

class DataGapError(ValueError):
    """Flags a gap in the available strain data.

    Runtime exception raised to communicate that a gap in the
    available strain data prevents further processing and plotting.
    """
    # pylint: disable-next=W0107
    pass

class ZeroFrequencyRangeError(ZeroDivisionError):
    """Flags a frequency range of zero width.

    Runtime exception raised upon detecting a frequency range
    whose lower and upper limits coincide, which is unsuitable for
    constructing a band pass filter.
    """
    # pylint: disable-next=W0107
    pass
