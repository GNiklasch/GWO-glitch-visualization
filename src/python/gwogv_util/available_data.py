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

"""Submodule for configuring and plotting usable/unusable data segments"""

from math import floor

import streamlit as st
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.ticker import NullFormatter, \
    AutoMinorLocator, NullLocator

# We implicitly use gwpy.timeseries.StateTimeSeries of which the flag data
# object will be an instance.

_lock = RendererAgg.lock

def configure(_app_conf, _appearance, _overrides):
    """Configure the module for use by a main application.

    Must be called before instantiating.  Various configurables are
    expected to be available as attributes of the three objects passed.
    """
    # pylint: disable=unused-argument
    global app_conf, appearance
    app_conf = _app_conf
    appearance = _appearance

class AvailableDataSegments:
    """Class for configuring and plotting data availability"""

    # This class has no object instantiation parameters and no
    # input choices to solicit.

    def plot_available_data_segments(
            self,
            data,
            data_descriptor,
            data_settings
    ) -> None:
        """Proceed to plot the available data segment(s)."""
        with _lock:
            figure_flag = data.flag_data.plot(
                figsize=appearance.AVAIL_FIGSIZE
            )
            ax = figure_flag.gca()
            ax.set_title(
                'Usable / unusable data vs. requested interval',
                fontsize=appearance.AVAIL_TITLE_FONTSIZE
            )
            ax.set_xscale('seconds', epoch=floor(data_settings.t0))
            if data_descriptor.t_end - data_descriptor.t_start == 128:
                # Major ticks appear every 15 seconds, subdivide accordingly:
                ax.xaxis.set_minor_locator(AutoMinorLocator(n=3))
            else:
                # The default subdivisions into 5 or 4 are fine for cache
                # wide blocks and for 96 s blocks.
                pass
            ax.yaxis.set_major_formatter(NullFormatter())
            ax.yaxis.set_major_locator(NullLocator())
            # Always highlight t0 in *this* diagram:
            ax.axvline(
                data_settings.t0,
                color=appearance.VLINE_COLOR,
                linestyle='--'
            )
            ax.axvline(
                data_settings.t_plotstart,
                color=appearance.PRIMARY_COLOR,
                linestyle='-.'
            )
            ax.axvline(
                data_settings.t_plotend,
                color=appearance.PRIMARY_COLOR,
                linestyle='-.'
            )
            st.pyplot(figure_flag, clear_figure=True)

