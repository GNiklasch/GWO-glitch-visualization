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

"""Submodule for configuring and plotting a constant-Q transform"""

from math import ceil

import streamlit as st
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
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

# ---------------------------------------------------------------------------
# -- Helper method (with cacheable results) --
# ---------------------------------------------------------------------------

# Recomputing a spectrogram or a const-Q transform doesn't take as long
# as plotting the results does, but caching them may still improve the
# user experience a little.
# Streamlit can't use the strain as the hash key, but it *can* use what
# we had used to fetch it - whence the dummy arguments.
@st.cache_data(max_entries=16 if large_caches else 4)
# pylint: disable-next=too-many-arguments, redefined-outer-name
def transform_strain(_strain, data_descriptor,
                     t_plotstart, t_plotend, t_pad, q_val, whiten):
    """Cacheable wrapper around TimeSeries.q_transform(), with graceful
    backoff to reduced padding when we're (too) close to a data gap"""
    # pylint: disable=unused-argument, redefined-outer-name
    outseg = (t_plotstart, t_plotend)
    # Without nailing down logf and fres, q_transform() would default to a
    # very high value for the number of frequency steps, somehow resulting
    # in exorbitant memory consumption for the ad-hoc modified colormaps
    # created during plotting  (on the order of 380 MiB for a single
    # Q-transform plot at high sample rate!).
    fres = ceil(
        max(600, 24 * q_val) * \
        (1 if data_descriptor.sample_rate < app_conf.HIGH_RATE \
         else 1.3)
    )
    q_warning = 0
    try:
        # The q_transform output would be distorted when the available strain
        # segment is much longer and isn't symmetric around t0:  Above a
        # certain frequency  (which depends on the padding and on the Q-value)
        # all features would be displaced to the left.  Pre-cropping the data
        # prevents this  (and speeds up processing, too).  Pre-cropping too
        # tightly, however, would result in broader whitening artefacts and
        # potentially losing output at low frequencies.
        padding = min(
            2.5 * t_pad,
            data_descriptor.t_end - t_plotend,
            t_plotstart - data_descriptor.t_start
        )
        strain_cropped = _strain.crop(
            t_plotstart - padding,
            t_plotend + padding
        )
        q_gram = strain_cropped.q_transform(
            outseg=outseg, qrange=(q_val, q_val),
            logf = True, fres = fres,
            whiten=whiten, fduration=t_pad
        )
    except ValueError:
        q_warning = 1
        try:
            # ...with less padding:
            strain_cropped = _strain.crop(
                t_plotstart - t_pad,
                t_plotend + t_pad
            )
            q_gram = strain_cropped.q_transform(
                outseg=outseg, qrange=(q_val, q_val),
                logf = True, fres = fres,
                whiten=whiten, fduration=t_pad
            )
        except ValueError:
            q_warning = 2
            # One last try, with no padding:
            strain_cropped = _strain.crop(t_plotstart, t_plotend)
            # Here, the default fduration=2 applies.
            q_gram = strain_cropped.q_transform(
                outseg=outseg, qrange=(q_val, q_val),
                logf = True, fres = fres,
                whiten=whiten
            )
    # If this last-ditch attempt fails, the exception is raised up
    # to our call site.
    return (q_gram, q_warning)

# ---------------------------------------------------------------------------
# -- Main class --
# ---------------------------------------------------------------------------

class QTransform:
    """Class for configuring and plotting a constant-Q transform"""

    def __init__(self, qtsf_settings):
        """Save settings dependent on the sample rate for later use."""
        self.qtsf_settings = qtsf_settings

    def solicit_choices(self) -> None:
        """Input widgets controlling how to plot the Q transform."""
        self.q0 = st.select_slider(
            '**Q-value:**',
            app_conf.Q_VALUES,
            value=app_conf.INITIAL_Q
        )
        self.ne_cutoff = st.select_slider(
            '**Normalized energy cutoff:**',
            app_conf.NORMALIZED_ENERGIES,
            value=app_conf.INITIAL_NE_CUTOFF
        )
        self.whiten_qtsf = st.checkbox(
            r'\- whiten before transforming',
            value=True
        )
        self.qtsf_grid_enabled = st.checkbox(
            r'\- enable grid overlay',
            value=True
        )
        self.qtsf_vline_enabled = st.checkbox(
            r'\- highlight t0',
            value=True
        )
        qtsf_colormap_choice = st.selectbox(
            '**Q transform colormap:**',
            appearance.COLORMAP_CHOICES,
            index=self.qtsf_settings.initial_colormap_choice
        )
        self.qtsf_colormap = appearance.COLORMAPS[qtsf_colormap_choice]

    def plot_q_transform(
            self,
            data,
            data_descriptor,
            data_settings
    ) -> None:
        """Proceed to plot the constant-Q transform.

        Relies on user input choices from a prior call to the
        `solicit_choices()` method.
        """
        q_wh_note = ', whitened' if self.whiten_qtsf else ''
        qtsf_title = (
            f'{data_descriptor.interferometer}, around'
            f' {data_settings.t0} ({data_settings.t0_iso} UTC),'
            f' Q={self.q0}{q_wh_note}'
        )
        try:
            q_gram, q_warning = transform_strain(
                data.strain,
                data_descriptor,
                data_settings.t_plotstart,
                data_settings.t_plotend,
                app_conf.T_PAD,
                q_val=self.q0,
                whiten=self.whiten_qtsf
            )
            q_error = False
        except ValueError:
            q_warning = 0
            q_error = True

        if q_error:
            st.error(
                '''t0 is too close to (or inside) a data gap, unable to
                compute the Q-transform. Try a shorter time interval or
                try varying the requested timestamp.'''
            )
            return

        with _lock:
            figure_qgram = q_gram.plot(figsize=self.qtsf_settings.figsize)
            ax = figure_qgram.gca()
            cax = make_axes_locatable(ax).append_axes(
                "right",
                size="5%", pad="3%"
            )
            figure_qgram.colorbar(
                label="Normalized energy",
                cax=cax, cmap=self.qtsf_colormap,
                clim=(0, self.ne_cutoff)
            )
            ax.set_title(qtsf_title, fontsize=appearance.QTSF_TITLE_FONTSIZE)
            ax.title.set_position([.5, 1.05])
            ax.grid(self.qtsf_grid_enabled)
            cax.grid(self.qtsf_grid_enabled)
            ax.set_xscale('seconds', epoch=data_settings.t_epoch)
            if data_settings.t_width >= 1.0:
                ax.xaxis.set_major_locator(
                    MultipleLocator(base=data_settings.t_major)
                )
            if data_settings.t_width <= 4.0:
                ax.xaxis.set_minor_locator(AutoMinorLocator(n=5))
            ax.set_yscale('log', base=2)
            ax.set_ylim(bottom=10)
            if self.qtsf_vline_enabled:
                ax.axvline(
                    data_settings.t0,
                    color=appearance.VLINE_COLOR,
                    linestyle='--'
                )
            st.pyplot(figure_qgram, clear_figure=True)

        if q_warning > 0:
            q_caveat = \
                '''t0 is close to a data gap, thus the Q-transform could
                not look{0} beyond the edges of what has been plotted and
                areas near these edges may contain artefacts.
                Also, information about low frequencies may be insufficient
                to paint that region.
                '''.format(' far' if q_warning == 1 else '')
            st.warning(q_caveat)
        else:
            pass
