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

"""Submodule for configuring and plotting a filtered TimeSeries"""

import streamlit as st
import numpy as np
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

from gwogv_util.exception import DataGapError

# We implicitly use gwpy.timeseries.TimeSeries of which the strain_cropped
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

class RawData:
    """Class for configuring and plotting a raw TimeSeries"""

    # This class has no object instantiation parameters.

    def solicit_choices(self) -> None:
        """Input widget controlling how to plot the raw data."""
        self.raw_vline_enabled = st.checkbox(
            r'\- highlight t0 in the raw data plot',
            value=True
        )

    def plot_raw_data(
            self,
            data,
            data_descriptor,
            data_settings
    ) -> None:
        """Proceed to plot the filtered data.

        Relies on one user input choice from a prior call to the
        `solicit_choices()` method.
        """

        raw_title = (
            f'{data_descriptor.interferometer}, around'
            f' {data_settings.t0} ({data_settings.t0_iso} UTC), raw'
        )

        # This might find no data to plot when the requested interval
        # falls inside a data gap.  The plot() method would fail silently
        # without raising an exception;  only the funny tick labels would
        # leave the viewer scratching their head.  But there's a slightly
        # obscure tell-tale:
        try:
            if np.isnan(data.strain_cropped.max().to_value()):
                raise DataGapError()

            with _lock:
                figure_raw = data.strain_cropped.plot(
                    color=appearance.PRIMARY_COLOR
                )
                ax = figure_raw.gca()
                ax.set_title(
                    raw_title,
                    loc='right',
                    fontsize=appearance.RAW_TITLE_FONTSIZE
                )
                ax.set_xscale('seconds', epoch=data_settings.t_epoch)
                if data_settings.t_width >= 1.0:
                    ax.xaxis.set_major_locator(
                        MultipleLocator(base=data_settings.t_major)
                    )
                if data_settings.t_width <= 4.0:
                    ax.xaxis.set_minor_locator(AutoMinorLocator(n=5))
                ax.set_ylabel('dimensionless')
                if self.raw_vline_enabled:
                    ax.axvline(
                        data_settings.t0,
                        color=appearance.VLINE_COLOR,
                        linestyle='--'
                    )
                st.pyplot(figure_raw, clear_figure=True)

        except DataGapError:
            st.error(
                '''t0 is too close to or inside a data gap. Please try
                a shorter time interval, or try changing the requested
                timestamp.'''
            )
            raise
