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

# pylint: disable=C0103,R0801,W0201
# pylint: disable=E0401
import streamlit as st
import numpy as np
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

from gwogv_util.exception import DataGapError, ZeroFrequencyRangeError

# We implicitly use gwpy.timeseries.TimeSeries of which the strain object
# will be an instance.

_lock = RendererAgg.lock

def configure(_app_conf, _appearance, _overrides):
    """Configure the module for use by a main application.

    Must be called before instantiating.  Various configurables are
    expected to be available as attributes of the three objects passed.
    """
    # pylint: disable=unused-argument
    # pylint: disable-next=W0601
    global app_conf, appearance
    app_conf = _app_conf
    appearance = _appearance

class FilteredData:
    """Class for configuring and plotting a filtered TimeSeries"""

    def __init__(self, filtered_settings):
        """Save settings dependent on the sample rate for later use."""
        self.filtered_settings = filtered_settings

    def solicit_choices(self) -> None:
        """Input widgets controlling how to plot the filtered data."""
        self.f_range = st.select_slider(
            '**Bandpass limits [Hz]:**',
            self.filtered_settings.f_detents_eff,
            value=self.filtered_settings.initial_f_range
        )
        self.whiten_plot = st.checkbox(
            r'\- whiten before filtering',
            value=True
        )
        self.filtered_vline_enabled = st.checkbox(
            r'\- highlight t0',
            value=True
        )

    def plot_filtered_data(
            self,
            data,
            data_descriptor,
            data_settings
    ) -> None:
        """Proceed to plot the filtered data.

        Relies on user input choices from a prior call to the
        `solicit_choices()` method.
        """
        strain_precropped = data.strain.crop(
            data_settings.t_plotstart - app_conf.T_PAD,
            data_settings.t_plotend + app_conf.T_PAD
        )
        wh_note = ', whitened' if self.whiten_plot else ''
        filtered_title = (
            f'{data_descriptor.interferometer}, around'
            f' {data_settings.t0} ({data_settings.t0_iso} UTC){wh_note},'
            f' band pass: {self.f_range[0]} - {self.f_range[1]} Hz'
        )
        filtered_y_label = 'arbitrary units' if self.whiten_plot \
            else 'dimensionless'
        try:
            if self.f_range[0] >= self.f_range[1]:
                # (Constructing the filter would raise a ValueError.)
                raise ZeroFrequencyRangeError()

            maybe_whitened = strain_precropped.whiten() if self.whiten_plot \
                else strain_precropped
            filtered = maybe_whitened.bandpass(
                self.f_range[0],
                self.f_range[1]
            )
            # Filtering will have failed when we are too close to a data gap,
            # and it fails silently - there's no exception we could catch.
            # But there's a tell-tale in the data:
            if np.isnan(filtered.max()):
                raise DataGapError()
            # In particular, there's no concern about the plot edge here.
            # If we didn't raise the DataGapError, we know we can plot the
            # one extra sample at the end.
            filtered_cropped = filtered.crop(
                data_settings.t_plotstart,
                data_settings.t_plotedge
            )

            with _lock:
                figure_filtered = filtered_cropped.plot(
                    color=appearance.PRIMARY_COLOR
                )
                ax = figure_filtered.gca()
                ax.set_title(
                    filtered_title,
                    loc='right',
                    fontsize=appearance.FILTERED_TITLE_FONTSIZE
                )
                ax.set_ylabel(filtered_y_label)
                ax.set_xscale('seconds', epoch=data_settings.t_epoch)
                if data_settings.t_width >= 1.0:
                    ax.xaxis.set_major_locator(
                        MultipleLocator(base=data_settings.t_major)
                    )
                if data_settings.t_width <= 4.0:
                    ax.xaxis.set_minor_locator(AutoMinorLocator(n=5))
                if self.filtered_vline_enabled:
                    ax.axvline(
                        data_settings.t0,
                        color=appearance.VLINE_COLOR,
                        linestyle='--'
                    )
                st.pyplot(figure_filtered, clear_figure=True)

                if self.f_range[0] < self.filtered_settings.calib_freq_low:
                    st.warning(self.filtered_settings.calib_caveat)
                else:
                    pass
        except DataGapError:
            st.error(
                '''t0 is too close to (or inside) a data gap, unable to
                filter the data. Please try a shorter time interval or
                try changing the requested timestamp.'''
            )
        except ZeroFrequencyRangeError:
            st.warning('Please make the frequency range wider.')
