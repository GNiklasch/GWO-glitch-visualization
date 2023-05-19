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

"""Submodule for configuring and plotting a spectrogram"""

import streamlit as st
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.ticker import NullFormatter, \
    AutoMinorLocator, MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

# We implicitly use gwpy.timeseries.TimeSeries of which the strain object
# will be an instance.

_lock = RendererAgg.lock

large_caches = False

def configure(_app_conf, _appearance, _overrides):
    """Configure the module for use by a main application.

    Must be called before instantiating.  Various configurables are
    expected to be available as attributes of the three objects passed.
    """
    global app_conf, appearance, large_caches
    app_conf = _app_conf
    appearance = _appearance
    large_caches = _overrides.large_caches

# Recomputing a spectrogram or a const-Q transform doesn't take as long
# as plotting the results does, but caching them may still improve the
# user experience a little.
# Streamlit can't use the strain as the hash key, but it *can* use what
# we had used to fetch and crop it - whence the dummy arguments.
@st.cache_data(max_entries=10 if large_caches else 4)
# pylint: disable-next=too-many-arguments, redefined-outer-name
def make_specgram(_strain, data_descriptor,
                  t_plotstart, t_plotend, stride, overlap):
    """Cacheable wrapper around TimeSeries.spectogram()"""
    # pylint: disable=unused-argument, redefined-outer-name
    specgram = _strain.spectrogram(
        stride=stride,
        overlap=overlap
    ) ** (1/2.)
    return specgram

class Spectrogram:
    """Class for configuring and plotting a spectrogram"""

    def __init__(self, spec_settings):
        """Save settings dependent on the sample rate for later use."""
        self.spec_settings = spec_settings

    def solicit_choices(self) -> None:
        """Input widgets controlling how to plot the spectrogram."""
        self.spec_f_range = st.select_slider(
            '**Spectrogram frequency range [Hz]:**',
            self.spec_settings.f_detents_eff,
            value=self.spec_settings.initial_f_range
        )
        spec_v_low, spec_v_high = st.select_slider(
            '**Spectrogram ASD range, decades:**',
            app_conf.SPEC_V_DETENTS,
            value=app_conf.SPEC_INITIAL_V
        )
        self.spec_v_min, self.spec_v_max = (
            app_conf.SPEC_V_DECADES[spec_v_low],
            app_conf.SPEC_V_DECADES[spec_v_high]
        )
        self.spec_grid_enabled = st.checkbox(
            r'\- enable grid overlay',
            value=True
        )
        self.spec_vline_enabled = st.checkbox(
            r'\- highlight t0',
            value=True
        )

        spec_colormap_choice = st.selectbox(
            '**Spectrogram colormap:**',
            appearance.COLORMAP_CHOICES,
            index=self.spec_settings.initial_colormap_choice
        )
        self.spec_colormap = appearance.COLORMAPS[spec_colormap_choice]

    def plot_spectrogram(
            self,
            data,
            data_descriptor,
            data_settings
    ) -> None:
        """Proceed to plot the spectrogram.

        Relies on user input choices from a prior call to the
        `solicit_choices()` method.
        """
        spec_title = (
            f'''{data_descriptor.interferometer}, around'''
            f''' {data_settings.t0} GPS ({data_settings.t0_iso} UTC)'''
        )
        spec_stride = min(
            data_settings.t_width / 8,
            app_conf.BASIC_SPEC_STRIDE
        )
        spec_overlap = spec_stride / 4
        specgram = make_specgram(
            data.strain_cropped,
            data_descriptor,
            data_settings.t_plotstart,
            data_settings.t_plotend,
            stride=spec_stride,
            overlap=spec_overlap
        )

        with _lock:
            figure_spec = specgram.plot(figsize=self.spec_settings.figsize)
            ax = figure_spec.gca()
            cax = make_axes_locatable(ax).append_axes(
                "right",
                size="5%", pad="3%"
            )
            figure_spec.colorbar(
                label=r'Strain ASD [${\mathrm{Hz}}^{-1/2}$]',
                cax=cax, cmap=self.spec_colormap,
                vmin=self.spec_v_min, vmax=self.spec_v_max,
                norm='log'
            )
            ax.set_title(
                spec_title,
                fontsize=appearance.SPEC_TITLE_FONTSIZE
            )
            cax.yaxis.set_minor_formatter(NullFormatter())
            ax.grid(self.spec_grid_enabled)
            cax.grid(self.spec_grid_enabled)
            ax.set_xscale('seconds', epoch=data_settings.t_epoch)
            if data_settings.t_width >= 1.0:
                ax.xaxis.set_major_locator(
                    MultipleLocator(base=data_settings.t_major)
                )
            if data_settings.t_width <= 4.0:
                ax.xaxis.set_minor_locator(AutoMinorLocator(n=5))
            ax.set_yscale('log', base=2)
            ax.set_ylim(self.spec_f_range)
            if self.spec_vline_enabled:
                ax.axvline(
                    data_settings.t0,
                    color=appearance.VLINE_COLOR,
                    linestyle='--'
                )
            st.pyplot(figure_spec, clear_figure=True)
