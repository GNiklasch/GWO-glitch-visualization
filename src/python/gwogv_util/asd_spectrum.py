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

"""Submodule for configuring and plotting amplitude spectral density"""

import streamlit as st
import numpy as np
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.ticker import LogFormatter, NullFormatter

from gwogv_util.exception import DataGapError
from gwogv_util.plotutil.ticker import MyFormatter

# We implicitly use gwpy.timeseries.TimeSeries of which the strain object
# will be an instance.

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

class ASDSpectrum:
    """Class for configuring and plotting an amplitude spectrum"""

    def __init__(self, asd_settings):
        """Save settings dependent on the sample rate for later use."""
        self.asd_settings = asd_settings

    def solicit_choices(self) -> None:
        """Input widgets controlling how to plot the ASD."""
        self.asd_f_range = st.select_slider(
            '**Spectrum frequency range [Hz]:**',
            self.asd_settings.f_detents_eff,
            value=self.asd_settings.initial_f_range
        )
        asd_y_low, asd_y_high = st.select_slider(
            '**Spectrum ASD range, decades:**',
            app_conf.ASD_Y_DETENTS,
            value=app_conf.ASD_INITIAL_Y
        )
        self.asd_y_range = (
            app_conf.ASD_Y_DECADES[asd_y_low],
            app_conf.ASD_Y_DECADES[asd_y_high]
        )
        asd_offset_choice = st.select_slider(
            '''**Also plot an optional background ASD spectrum,
            from [seconds] earlier or later:**''',
            app_conf.ASD_OFFSET_DETENTS,
            value=app_conf.ASD_INITIAL_OFFSET
        )
        self.asd_offset = app_conf.ASD_OFFSETS[asd_offset_choice]
        # Just in case someone wants to extract a light-shaded plot:
        self.asd_lighten = st.checkbox(
            r'\- swap shades: light foreground (and heavy background)'
        )

    def plot_asd_spectrum(
            self,
            data,
            data_descriptor,
            data_settings
    ) -> None:
        """Proceed to plot the Amplitude Spectral Density.

        Relies on user input choices from a prior call to the
        `solicit_choices()` method.
        """
        asd_title = (
            f'''{data_descriptor.interferometer},'''
            f''' during {data_settings.t_width} s around'''
            f''' {data_settings.t0} GPS ({data_settings.t0_iso} UTC)'''
        )
        asd_xlabel = (
            f'Frequency [Hz],'
            f' {self.asd_f_range[0]} - {self.asd_f_range[1]} Hz'
        )
        asd_ylabel = r'Strain ASD [${\mathrm{Hz}}^{-1/2}$]'
        asd_bgnd_warning = False

        try:
            # Computing the ASD will fail when the cropped interval overlaps
            # a data gap.  The tell-tale symptom is somewhat obscure:
            strain_asd = data.strain_cropped.asd()
            if np.isnan(strain_asd.max().to_value()):
                raise DataGapError()

            if not self.asd_offset == 0:
                strain_bgnd_asd = data.strain.crop(
                    data_settings.t_plotstart + self.asd_offset,
                    data_settings.t_plotend + self.asd_offset
                ).asd()
                if np.isnan(strain_bgnd_asd.max().to_value()):
                    asd_bgnd_warning = True
                else:
                    if self.asd_offset > 0:
                        asd_bgnd_label = f'{self.asd_offset} s later'
                    else:
                        asd_bgnd_label = f'{-self.asd_offset} s earlier'
            else:
                pass

            with _lock:
                figure_asd = strain_asd.plot(
                    figsize=appearance.ASD_FIGSIZE,
                    color=appearance.ASD_LIGHT_COLOR if self.asd_lighten \
                    else appearance.PRIMARY_COLOR
                )
                ax = figure_asd.gca()
                # Now that we have the axes configured, plotting a
                # non-existent background FrequencySeries would just
                # fail silently - but no need to even try when we
                # already know it wouldn't work.
                if not self.asd_offset == 0 and not asd_bgnd_warning:
                    ax.plot(
                        strain_bgnd_asd,
                        label=asd_bgnd_label,
                        color=appearance.PRIMARY_COLOR if self.asd_lighten \
                        else appearance.ASD_TRANSPARENT_COLOR
                    )
                    # We'll let matplotlib pick the best corner for the
                    # legend.
                    # GWpy's custom handler_map creates an example line
                    # segment that's rather thick, but with handler_map=None
                    # to reinstate matplotlib's defaults it would be too thin.
                    ax.legend(fontsize=appearance.ASD_TITLE_FONTSIZE)
                ax.set_title(
                    asd_title,
                    fontsize=appearance.ASD_TITLE_FONTSIZE,
                    loc='right', pad=10.
                )
                ax.xaxis.set_major_formatter(LogFormatter(base=10))
                ax.xaxis.set_minor_formatter(MyFormatter(self.asd_f_range))
                ax.yaxis.set_minor_formatter(NullFormatter())
                ax.set_xlim(self.asd_f_range)
                ax.set_ylim(self.asd_y_range)
                ax.set_ylabel(
                    asd_ylabel,
                    fontsize=appearance.ASD_LABEL_FONTSIZE
                )
                ax.set_xlabel(
                    asd_xlabel,
                    fontsize=appearance.ASD_LABEL_FONTSIZE
                )
                ax.xaxis.set_tick_params(
                    which='major',
                    labelsize=appearance.ASD_LABEL_FONTSIZE
                )
                ax.xaxis.set_tick_params(
                    which='minor',
                    labelsize=appearance.ASD_LABEL_LABELSIZE
                )
                ax.yaxis.set_tick_params(
                    which='major',
                    labelsize=appearance.ASD_LABEL_LABELSIZE
                )
                st.pyplot(figure_asd, clear_figure=True)

                if self.asd_f_range[0] < self.asd_settings.calib_freq_low:
                    st.warning(self.asd_settings.calib_caveat)

                if asd_bgnd_warning:
                    st.warning(
                        '''t0 is too close to a data gap, unable to include
                        a background spectrum. Try changing the time offset.'''
                    )

        except DataGapError:
            # Just in case the caller hasn't noticed this before and
            # stopped processing before calling us...
            st.error(
                '''t0 is too close to (or inside) a data gap, unable to
                extract a spectrum. Try a shorter time interval or try
                varying the requested timestamp.'''
            )
