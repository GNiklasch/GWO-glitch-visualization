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

"""Submodule for loading a TimeSeries of strain data from GWOSC"""

# pylint: disable=C0103,W0601,W0612,W0621,R0801
# pylint: disable=E0401
import streamlit as st
import numpy as np
from gwpy.timeseries import TimeSeries, StateTimeSeries

large_caches = False
url_caching = False

def configure(_app_conf, _appearance, _overrides):
    """Configure the module for use by a main application.

    Must be called before instantiating.  Various configurables are
    expected to be available as attributes of the three objects passed.
    """
    # pylint: disable-next=W0601
    global app_conf, appearance, large_caches
    app_conf = _app_conf
    appearance = _appearance
    large_caches = _overrides.large_caches
    url_caching = _overrides.url_caching

# ---------------------------------------------------------------------------
# -- Loader methods (with cacheable results) --
# ---------------------------------------------------------------------------

def _load_strain_impl(data_descriptor):
    """Workhorse wrapper around TimeSeries.fetch_open_data()"""
    # pylint: disable=redefined-outer-name
    # Work around bug #1612 in GWpy:  fetch_open_data() would fail if t_end
    # falls on  (or a fraction of a second before)  the boundary between
    # two successive 4096 s chunks.  Asking for a fraction of a second
    # *more* just past this boundary avoids the issue.
    t_end_fudged = data_descriptor.t_end + 1/64.
    # (Is GWpy's URL caching thread-safe?  I certainly don't dare to turn
    # it on for multi-user operation in the cloud, without having control
    # over cache entry lifetimes.)
    # The other question is whether it's actually useful here...
    strain = TimeSeries.fetch_open_data(
        data_descriptor.interferometer,
        data_descriptor.t_start,
        t_end_fudged,
        sample_rate=data_descriptor.sample_rate,
        cache=url_caching)
    # Extract some information about the available vs. unavailable data
    # in the vicinity, based on our own inspection of what we got from GWOSC
    # (rather than expending yet more time to fetch various metadata):
    intervals = \
        int((data_descriptor.t_end - data_descriptor.t_start) * 8) - 1
    flag = StateTimeSeries(
        [
            not np.isnan(
                strain.value_at(data_descriptor.t_start + i/8. + 1/16.)
            )
            for i in range(intervals)
        ],
        sample_rate=8,
        epoch=data_descriptor.t_start
    ).to_dqflag(round=False)
    return (strain, flag)

# The load_strain() indirection below will branch to one or the other
# wrapper, because we want separately sized caches for low and high
# sample rate data.
# (With our time interval padding, caching 128 s at the low sample rate
# amounts to a cache usage of roughly 20 MiB, plus some internal overhead,
# and a high rate cache item to four times as much.)
# For local use where more than 1 GiB of RAM is available, we may use wider
# cache blocks  (typically 512 s at the low sample rate)  when requested.
@st.cache_data(max_entries=16 if large_caches else 8)
def load_low_rate_strain(data_descriptor):
    """Cacheable wrapper around low-sample-rate data fetching"""
    # pylint: disable=redefined-outer-name
    return _load_strain_impl(data_descriptor)

@st.cache_data(max_entries=8 if large_caches else 3)
def load_high_rate_strain(data_descriptor):
    """Cacheable wrapper around high-sample-rate data fetching"""
    # pylint: disable=redefined-outer-name
    return _load_strain_impl(data_descriptor)
