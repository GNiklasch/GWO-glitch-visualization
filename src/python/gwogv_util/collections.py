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

"""Classes encapsulating collections of attributes"""

from collections import namedtuple

class AttributeHolder:
    """Instances of this class will serve as glorified dictionaries."""
    # pylint: disable-next=W0107
    pass

DataDescriptor = namedtuple(
    'DataDescriptor',
    [
        'interferometer',
        't_start',
        't_end',
        'sample_rate'
    ]
)

DataDescriptor.__doc__ = \
    """Parameters identifying the loaded strain data segment.

    This provides a hashable, immutable tuple whose fields are accessible
    through attribute references.
    """

# pylint: disable=R0902, R0903
