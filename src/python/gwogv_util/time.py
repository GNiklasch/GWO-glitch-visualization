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

"""Utilites for GPS / UTC timestamp handling"""

import astropy.time as atime

# ---------------------------------------------------------------------------
# -- Timestamp conversion --
# ---------------------------------------------------------------------------

# Unlike `datetime`, `astropy` treats leap seconds correctly.

def gps_to_isot(val):
    """Convert a GPS timestamp to a UTC date/time in ISO 8601 format with
    a literal 'T' separating date and time."""
    # pylint: disable-next=redefined-builtin
    return atime.Time(
        val=atime.Time(val=val, scale='tai', format='gps'),
        scale='utc', format='isot'
    ).to_string()

# pylint: disable-next=redefined-builtin
def iso_to_gps(val, format='isot'):
    """Convert a UTC date/time in ISO 8601 format with a literal 'T'
    separating date and time to a GPS timestamp."""
    # pylint: disable-next=redefined-builtin
    return atime.Time(
        val=atime.Time(val=val, scale='utc', format=format),
        scale='tai', format='gps'
    ).to_value('gps')

def any_to_gps(val):
    """Convert the user intput to a GPS timestamp, accepting either
    UTC formatted as ISO 8601 date/time with 'T' or space separating
    time and date, optionally with a trailing 'Z', or text that can be
    parsed as a floating point number representing a GPS timestamp."""
    try:
        t_gps = iso_to_gps(val=val)
    # pylint: disable-next=broad-exception-caught
    except Exception:
        try:
            # pylint: disable-next=redefined-builtin
            t_gps = iso_to_gps(val=val, format='iso')
        # pylint: disable-next=broad-exception-caught
        except Exception:
            t_gps = float(val)
    return t_gps

# ---------------------------------------------------------------------------
# -- Current time as ISO 8601 --
# ---------------------------------------------------------------------------

def now_as_isot():
    return atime.Time(
        atime.Time.now(),
        scale='utc', format='isot'
    ).to_string()
