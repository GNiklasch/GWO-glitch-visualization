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

"""An interactive utility for visualizing public strain data from GWOSC.

It is capable of plotting raw strain time series data,
whitened and bandpass-filtered views of these time series data,
spectra showing their amplitude spectral density during a time interval,
spectrograms showing how the ASD evolves during this interval, and/or
constant-Q transformed "spectrograms" adapted to visualizing transient
features, notably glitches (although some real GW events also show up).

This is intended to be used for investigating glitches in LIGO and Virgo
strain data, in conjunction with the Gravity Spy project on Zooniverse
and with either the Gravity Spy tools hosted by CIERA/NWU or the Gravity
Spy data release files as sources of relevant timestamps.

It may also become useful to the GWitchHunters project on Zooniverse
if that project should provide access to event timestamps one day.

All user interface controls are grouped into five forms, each with its
own submit button, in a sidebar to the left of the main pane hosting
the graphics.
"""

# See note at end about Pylint (as a GitHub workflow).
# pylint: disable=C0103, C0209

# ---------------------------------------------------------------------------
# -- Imports and matplotlib backend sanitizing --
# ---------------------------------------------------------------------------

from math import ceil, floor
import argparse

# memory management and profiling
import gc
import tracemalloc

# pylint: disable=E0401
import streamlit as st
import numpy as np
from gwpy.timeseries import TimeSeries, StateTimeSeries
import matplotlib as mpl
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.ticker import LogFormatter, NullFormatter, \
    AutoMinorLocator, MultipleLocator, NullLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

from gwogv_util.collections import AttributeHolder, DataDescriptor
from gwogv_util.exception import DataGapError
from gwogv_util.time import gps_to_isot, iso_to_gps, any_to_gps, now_as_isot
import gwogv_util.available_data as available_gram
import gwogv_util.filtered_data as filtered_gram
import gwogv_util.spectrogram as spec_gram
import gwogv_util.q_transform as qtsf_gram
import gwogv_util.asd_spectrum as asd_gram

# Importing customcm is required since it registers our custom colormap
# and its reversed form with matplotlib.
# pylint: disable-next=unused-import
import gwogv_util.plotutil.customcm

# Thread-safe plotting:
# The backend choice here is redundant - Streamlit already does this.
mpl.use("agg")
_lock = RendererAgg.lock

# ---------------------------------------------------------------------------
# -- Commandline override switches --
# ---------------------------------------------------------------------------

# Several sizing defaults below are coded with a view to deploying this
# application into a RAM-limited Streamlit Community Cloud container.
# For local use where more RAM is available, more generous settings can
# be unlocked by commandline options:
parser = argparse.ArgumentParser(prog='glitchplot')
parser.add_argument('-C', action='store_true', dest='large_caches')
parser.add_argument('-M', action='store_true', dest='mem_profiling')
parser.add_argument('-U', action='store_true', dest='url_caching')
parser.add_argument('-W', action='store_true', dest='wide_cache_blocks')
parser.add_argument('-s', action='store_true', dest='silence_notices')
overrides = parser.parse_args()

# (An alternative would have been to (ab)use .streamlit/secrets.toml to
# hold cloud-specific configuration settings.  They could then be tuned
# on the fly without a GitHub commit and redeployment.  But we'd still
# need *some* usable defaults here to enable people to run the code locally
# without creating a secrets.toml file, so this would not gain much.)

# ---------------------------------------------------------------------------
# -- Input selectables and related parameters, up front --
# ---------------------------------------------------------------------------

app_conf = AttributeHolder()
app_conf.LOW_RATE, app_conf.HIGH_RATE = (4096, 16384)

# ---------------------------------------------------------------------------
# -- Helper methods: cacheable data...
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
        cache=overrides.url_caching)
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
@st.cache_data(max_entries=16 if overrides.large_caches else 8)
def load_low_rate_strain(data_descriptor):
    """Cacheable wrapper around low-sample-rate data fetching"""
    # pylint: disable=redefined-outer-name
    return _load_strain_impl(data_descriptor)

@st.cache_data(max_entries=8 if overrides.large_caches else 3)
def load_high_rate_strain(data_descriptor):
    """Cacheable wrapper around high-sample-rate data fetching"""
    # pylint: disable=redefined-outer-name
    return _load_strain_impl(data_descriptor)

# ---------------------------------------------------------------------------
# ...memory profiling...
# ---------------------------------------------------------------------------

def print_mem_profile(tops = 8) -> None:
    """Print out some memory diagnostics."""
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')
    print(f'[- Top {tops} -]')
    for stat in top_stats[:tops]:
        print(stat)
    print('----------------')

# ---------------------------------------------------------------------------
# ...page footer --
# ---------------------------------------------------------------------------

def emit_footer() -> None:
    """Emit the page footer."""
    st.divider()
    # pylint: disable=anomalous-backslash-in-string
    footer = '''
    View the source code on [GitHub]({0}).\\
    Inspired by [GW Quickview]({1}).\\
    Powered by [GWpy]({2}); fed with [data]({3}) hosted by the [GWOSC]({4}).\\
    Web user interface created with [Streamlit]({5}).\\
    \- Page refreshed {6} UTC.
    '''.format(
        'https://github.com/GNiklasch/GWO-glitch-visualization',
        'https://github.com/jkanner/streamlit-dataview/',
        'https://gwpy.github.io/',
        'https://gwosc.org/data/',
        'https://gwosc.org/',
        'https://streamlit.io/',
        now_as_isot()
    )
    st.markdown(footer)
    if not overrides.silence_notices:
        cloud_notice = '''
        This application neither expects, nor processes, nor stores any
        personal data. The Streamlit Community Cloud hosting platform
        on which it is deployed, like any responsible web service provider,
        is aware of your IP address and logs accesses, and it collects some
        anonymized usage statistics - view their [privacy policy]({0}).
        '''.format(
            'https://streamlit.io/privacy-policy'
        )
        st.caption(cloud_notice)

# ---------------------------------------------------------------------------
# -- Appearance:  Page layout...
# ---------------------------------------------------------------------------

APPTITLE = 'GWO glitch plotter'
st.set_page_config(
    page_title=APPTITLE,
    page_icon=":sparkler:"
)

# CSS tweaks:  Hide Streamlit's own version of the footer, and get rid
# of excessive vertical padding while we're at it.
# This must be emitted right at the start  (before emitting the title!)
# in order to have the page look right immediately when it is visited.
HIDE_ST_FOOTER = '''
<style>
footer {visibility: hidden;}

.appview-container > section:first-child > div:first-child > div:nth-child(2) {
  padding-top: 1rem;
}

.block-container {
  padding-top: 2.5rem;
  padding-bottom: 1rem;
}
</style>
'''
st.markdown(HIDE_ST_FOOTER, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# ...colors and colormaps...
# ---------------------------------------------------------------------------

appearance = AttributeHolder()

appearance.PRIMARY_COLOR = st.get_option("theme.primaryColor") # '#0F2CA4'
appearance.VLINE_COLOR = 'orange'

# Both of the following are derived from PRIMARY_COLOR and will produce
# very nearly the same shade when plotted on a white background.  The
# transparent version becomes invisible when plotted over the PRIMARY_COLOR,
# making it possible to plot a background curve after plotting the foreground
# curve.
# (If one wants the background curve to be faintly visible "through" the
# foreground one, the transparent color shade could be made slightly lighter
# and slightly more opaque, interpolating between the current two specs.)
appearance.ASD_LIGHT_COLOR = '#7A89C8'
appearance.ASD_TRANSPARENT_COLOR = \
    appearance.PRIMARY_COLOR + '96' # '#0F2CA496'

appearance.COLORMAPS = {
    'Viridis (Gravity Spy)': 'viridis',
    'Viridis reversed': 'viridis_r',
    'Reds': 'Reds',
    'Reds reversed (GwitchHunters)': 'Reds_r',
    'Blues': 'Blues',
    'Blues reversed': 'Blues_r',
    'Plasma': 'plasma',
    'Plasma reversed': 'plasma_r',
    'Cividis': 'cividis',
    'Cividis reversed': 'cividis_r',
    'Cubehelix': 'cubehelix',
    'Cubehelix reversed': 'cubehelix_r',
    'Jetstream': 'jetstream',
    'Jetstream reversed': 'jetstream_r'
}
appearance.COLORMAP_CHOICES = list(appearance.COLORMAPS)

# ---------------------------------------------------------------------------
# ...figure and font sizes...
# ---------------------------------------------------------------------------

appearance.AVAIL_FIGSIZE = (12, 0.6)
appearance.ASD_FIGSIZE = (10, 8)

appearance.AVAIL_TITLE_FONTSIZE = 14
appearance.RAW_TITLE_FONTSIZE = 14
appearance.FILTERED_TITLE_FONTSIZE = 14
appearance.ASD_TITLE_FONTSIZE = 14
appearance.ASD_LABEL_FONTSIZE = 13
appearance.ASD_LABEL_LABELSIZE = 11
appearance.SPEC_TITLE_FONTSIZE = 17
appearance.QTSF_TITLE_FONTSIZE = 17

# ---------------------------------------------------------------------------
# ...and the (in-page) title:
# ---------------------------------------------------------------------------

st.title('Plot glitches from GWOSC-sourced strain data')

# ---------------------------------------------------------------------------
# -- Input selectables and related parameters, remainder --
# ---------------------------------------------------------------------------

# Strictly speaking the following is valid for O3 only...
app_conf.calib_freqs_low = {'H1': 10, 'L1': 10, 'V1': 20}
app_conf.interferometers = list(app_conf.calib_freqs_low)

# The L1 view of GW170817, with the almighty ETMY saturation / ESD overflow
# glitch a second earlier, is well known as the Gravity Spy logo.
app_conf.INITIAL_T0_GPS = '1187008882.4'

# Loading time is dominated by having to wade through either one or two
# files of 4096 seconds of strain data.  Asking for a generously sized
# time interval of a couple of minutes does not add to the latency,
# except when it requires two files to be fetched instead of just one.
# It does come with a memory penalty, as noted above.
# Making the elbow room substantially larger than half the maximal interval
# width we're going to plot avoids unnecessary cache-miss reloads when
# zooming in or out to shorter or longer time ranges.  The particular
# choice of T_ELBOW_ROOM ensures that we'll always load either 96 or 128
# seconds' worth when not using optional wide cache blocks locally.  It is
# also tied to that of INITIAL_T0_GPS:  Panning 2 or 3 seconds into the past
# or future from this timestamp should not push either edge over a 32 s
# boundary.  (Of course this can't always be true;  jiggling a random t0
# around by 2 or 3 seconds *can* cause a cache miss.  But I specifically
# wanted to avoid this for the GW170817 event timestamp shown at startup.)
# Also, we'll be able to extract a "background" spectrum from 64s worth of
# data cropped at time offsets up to ±12 seconds  (useless as this is when
# the half width is larger than the offset)  without falling off the ends.
app_conf.T_ELBOW_ROOM = 46.7

# Filtering will have to get by with less padding beyond the half width:
app_conf.T_PAD = 8.0

app_conf.T_WIDTHS = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8, 16, 32, 64)
app_conf.INITIAL_WIDTH = app_conf.T_WIDTHS[5]

app_conf.SAMPLE_RATES = (app_conf.LOW_RATE, app_conf.HIGH_RATE)

app_conf.CHUNK_SIZE = 4096 # only used for informative messages

# It might be nice to have a red "Don't...." and green "Do...", but Streamlit
# selectbox options do not support markdown/colors.  Moreover, selecting any
# particular selectbox widget for CSS manipulations is awkward at best, and
# having the CSS settings depend on the textual content of the <div> element
# is not practical.
# (Radio button options don't support markdown either;  and radio buttons
# would take up too much space, as well as breaking the flow of the sentence.)
app_conf.YORN = {"Don't...": False, 'Do...': True}
app_conf.YORN_CHOICES = list(app_conf.YORN)

# Usable frequencies for filtering depend on the sample rate.
# Thus the following will be sliced down to size as appropriate - we need
# to stay well below the Nyquist limit in order for the filter to be built
# correctly.  ASD plot boundaries can use one more slot.  Plotting up to
# 2048 resp. 8192 Hz  (at low resp. high sample rate)  would work but would
# produce a substantial bogus downturn where the anti-aliasing pre-filter
# kicks in.
# Data below calib_freqs_low[interferometer] are also unreliable.
# (H1 in particular switched to much more aggressive VLF highpass filtering
# during O3b on 2020-01-14.)
app_conf.F_DETENTS = (
    8, 9.51, 11.3, 13.5, 16, 19.0, 22.6, 26.9,
    32, 38.3, 45.3, 53.8, 64, 76.1, 90.5, 108,
    128, 152, 181, 215, 256, 304, 362, 431,
    512, 609, 724, 861, 1024, 1218, 1448, 1722,
    2048, 2435, 2896, 3444, 4096, 4871, 5793
)

# ASD spectra and spectrograms share some of their input choices.
app_conf.ASD_Y_DECADES = {
    -27: 1.E-27, -26: 1.E-26, -25: 1.E-25, -24: 1.E-24,
    -23: 1.E-23, -22: 1.E-22, -21: 1.E-21, -20: 1.E-20,
    -19: 1.E-19, -18: 1.E-18, -17: 1.E-17, -16: 1.E-16
}
app_conf.ASD_Y_DETENTS = list(app_conf.ASD_Y_DECADES)
app_conf.ASD_OFFSETS = {
    '-12': -12, '-6': -6, '-3': -3, '-1.5': -1.5,
    'None': 0,
    '1.5': 1.5, '3': 3, '6': 6, '12': 12
}
app_conf.ASD_OFFSET_DETENTS = list(app_conf.ASD_OFFSETS)
app_conf.ASD_INITIAL_OFFSET = app_conf.ASD_OFFSET_DETENTS[4] # 'None'
app_conf.ASD_INITIAL_Y = (
    app_conf.ASD_Y_DETENTS[1],
    app_conf.ASD_Y_DETENTS[-4]
)

app_conf.SPEC_V_DECADES = app_conf.ASD_Y_DECADES
app_conf.SPEC_V_DETENTS = list(app_conf.SPEC_V_DECADES)
app_conf.SPEC_INITIAL_V = (
    app_conf.SPEC_V_DETENTS[1],
    app_conf.SPEC_V_DETENTS[-4]
)

# Spectrograms will have 8 Hz resolution except for very short widths where
# the stride will have to be adjusted down.  4 Hz (0.25 s stride) is too fine;
# the LIGO fundamental violin modes would then get lost under the horizontal
# grid lines.
app_conf.BASIC_SPEC_STRIDE = 0.125

# View limit frequencies for spectrograms:
app_conf.SPEC_F_DETENTS = (
    10, 22.6, 45.3, 90.5, 181, 362, 724, 1448, 2896, 5793
)

# Q-values spaced at powers of sqrt(2):
app_conf.Q_VALUES = (5.66, 8, 11.3, 16, 22.6, 32, 45.3, 64)
app_conf.INITIAL_Q = app_conf.Q_VALUES[2]

# Colormap scaling for const-Q transforms:
app_conf.NORMALIZED_ENERGIES = (6.3, 12.7, 25.5, 51.1, 102.3)
app_conf.INITIAL_NE_CUTOFF = app_conf.NORMALIZED_ENERGIES[2]

# Start memory profiling if requested:
if overrides.mem_profiling:
    tracemalloc.start()

# ---------------------------------------------------------------------------
# -- Command and control sidebar --
# ---------------------------------------------------------------------------

st.sidebar.markdown('# Controls')

st.sidebar.write(
    '''In each panel below, make adjustments as desired,
    then click the button underneath to apply them.'''
)

# ... data-loading form:

available_gram.configure(app_conf, appearance, overrides)
available_plotter = available_gram.AvailableDataSegments()

with st.sidebar.form('load_what'):

    if not overrides.silence_notices and \
       'silenced' not in st.session_state:
        st.caption(
            '''This appplication uses essential cookies to ensure
            consistent behavior from one button click to the next.
            It won't remember you from one visit to the next, so this
            notice will keep reappearing.
            The cloud hosting platform also uses cookies to collect
            anonymized usage statistics.'''
        )
        st.session_state['silenced'] = st.checkbox(
            r'\- Got it!',
            value=False
        )

    st.markdown('### Load what from where:')

    interferometer = st.selectbox(
        '**From interferometer:**',
        app_conf.interferometers,
        index=1 # default L1
    )
    t_width = st.select_slider(
        '**load enough to visualize**',
        app_conf.T_WIDTHS, value=app_conf.INITIAL_WIDTH
    )
    t0_text = st.text_input(
        '''**seconds of strain data around GPS
        (or ISO 8601-formatted UTC) timestamp t0:**''',
        app_conf.INITIAL_T0_GPS
    )
    cache_wide_blocks = False
    if overrides.wide_cache_blocks:
        cache_wide_blocks = st.checkbox(
            r'\- use wide cache blocks',
            value=True
        )
    raw_vline_enabled = st.checkbox(
        r'\- highlight t0 in the raw data plot',
        value=True
    )
    sample_rate = st.selectbox(
        '**Sample rate:**',
        app_conf.SAMPLE_RATES
    )

    load_submitted = st.form_submit_button(
        'Load and plot raw data',
        type='primary',
        use_container_width=True
    )

# Preprocess parameters which depend on the sample rate and/or interferometer:

filtered_settings = AttributeHolder()
asd_settings = AttributeHolder()
spec_settings = AttributeHolder()
qtsf_settings = AttributeHolder()

if sample_rate < app_conf.HIGH_RATE:
    filtered_settings.f_detents_eff = app_conf.F_DETENTS[0:30]
    asd_settings.f_detents_eff = app_conf.F_DETENTS[0:31]
    spec_settings.f_detents_eff = app_conf.SPEC_F_DETENTS[0:8]
    spec_settings.figsize = (12, 6)
    qtsf_settings.figsize = (12, 7)
    load_strain = load_low_rate_strain
else:
    filtered_settings.f_detents_eff = app_conf.F_DETENTS[0:38]
    asd_settings.f_detents_eff = app_conf.F_DETENTS
    spec_settings.f_detents_eff = app_conf.SPEC_F_DETENTS
    spec_settings.figsize = (12, 7)
    qtsf_settings.figsize = (12, 8)
    load_strain = load_high_rate_strain

filtered_settings.initial_f_range = (
    filtered_settings.f_detents_eff[1],
    filtered_settings.f_detents_eff[28]
)
asd_settings.initial_f_range = (
    asd_settings.f_detents_eff[1],
    asd_settings.f_detents_eff[-1]
)
spec_settings.initial_f_range = (
    spec_settings.f_detents_eff[0],
    spec_settings.f_detents_eff[-1]
)

# Default for spectrograms is our custom Jetstream colormap, which is similar
# to one used in the Virgo electronic logs.
# Streamlit doesn't allow negative indices counting backward from
# the end of the selectbox options...
spec_settings.initial_colormap_choice = len(appearance.COLORMAP_CHOICES)-2
# Default for the Q transform is Viridis (same as Gravity Spy's).
qtsf_settings.initial_colormap_choice = 0

calib_freq_low = app_conf.calib_freqs_low[interferometer]
calib_caveat = (
    f'Caution: Strain data below {calib_freq_low} Hz from'
    f" {interferometer} aren't calibrated."
)
filtered_settings.calib_freq_low = calib_freq_low
filtered_settings.calib_caveat = calib_caveat
asd_settings.calib_freq_low = calib_freq_low
asd_settings.calib_caveat = calib_caveat

# pylint: disable-next=implicit-str-concat
st.sidebar.caption(
    (''' Use the "Do..." and "Don't..." options to select'''
     ' which plots to show or skip.')
)

# ... filtered-plot form:

filtered_gram.configure(app_conf, appearance, overrides)
filtered_plotter = filtered_gram.FilteredData(filtered_settings)

with st.sidebar.form('plot_how'):
    # Default here is "Don't".
    do_plot_txt = st.selectbox(
        'Shall we plot?',
        app_conf.YORN_CHOICES,
        label_visibility = 'collapsed'
    )
    do_plot = app_conf.YORN[do_plot_txt]

    st.markdown('### ...filter and plot filtered data:')
    filtered_plotter.solicit_choices()

    plot_submitted = st.form_submit_button(
        'Apply filtered plot settings',
        type='primary',
        use_container_width=True
    )

# ... ASD spectrum form:

asd_gram.configure(app_conf, appearance, overrides)
asd_plotter = asd_gram.ASDSpectrum(asd_settings)

with st.sidebar.form('asd_how'):
    do_show_asd_txt = st.selectbox(
        'Shall we show ASD?',
        app_conf.YORN_CHOICES, # Default is "Don't".
        label_visibility = 'collapsed'
    )
    do_show_asd = app_conf.YORN[do_show_asd_txt]

    st.markdown('### ...show amplitude spectral density as a spectrum:')
    asd_plotter.solicit_choices()

    asd_submitted = st.form_submit_button(
        'Apply spectrum settings',
        type='primary',
        use_container_width=True
    )

# ... Spectrogram form:

spec_gram.configure(app_conf, appearance, overrides)
spec_plotter = spec_gram.Spectrogram(spec_settings)

with st.sidebar.form('spec_how'):
    do_spec_txt = st.selectbox(
        'Shall we spec?',
        app_conf.YORN_CHOICES, # Default here is "Don't".
        label_visibility = 'collapsed'
    )
    do_spec = app_conf.YORN[do_spec_txt]

    st.markdown('### ...show a spectrogram:')
    spec_plotter.solicit_choices()

    spec_submitted = st.form_submit_button(
        'Apply spectrogram settings',
        type='primary',
        use_container_width=True
    )

# ... Q-transform form:

qtsf_gram.configure(app_conf, appearance, overrides)
qtsf_plotter = qtsf_gram.QTransform(qtsf_settings)

with st.sidebar.form('qtsf_how'):
    do_qtsf_txt = st.selectbox(
        'Shall we qtsf?',
        app_conf.YORN_CHOICES,
        label_visibility = 'collapsed',
        index=1 # Default here is 'Do'.
    )
    do_qtsf = app_conf.YORN[do_qtsf_txt]

    st.markdown('### ...render a constant-Q transform:')
    qtsf_plotter.solicit_choices()


    qtsf_submitted = st.form_submit_button(
        'Apply Q transform settings',
        type='primary',
        use_container_width=True
    )

# ---------------------------------------------------------------------------
# -- Data load preparations --
# ---------------------------------------------------------------------------

try:
    # pylint: disable-next=bad-str-strip-call
    t0=any_to_gps(t0_text.strip())
except ValueError as ex:
    st.warning('Sorry, there seems to be a typo in the timestamp input:')
    st.error(ex)
    st.warning('Please correct and re-submit your load request.')
    emit_footer()
    st.stop()

t0_iso=gps_to_isot(t0)

t_cache_boundaries = (
    (512 if sample_rate < app_conf.HIGH_RATE else 256) \
    if cache_wide_blocks else 32
)

t_halfwidth = t_width / 2
t_start = t_cache_boundaries * \
    floor((t0 - app_conf.T_ELBOW_ROOM)/t_cache_boundaries)
t_end = t_cache_boundaries * \
    ceil((t0 + app_conf.T_ELBOW_ROOM)/t_cache_boundaries)
t_plotstart = t0 - t_halfwidth
t_plotend = t0 + t_halfwidth

data_descriptor = DataDescriptor(
    interferometer=interferometer,
    t_start=t_start,
    t_end=t_end,
    sample_rate=sample_rate
)

# GWpy's GPSLocator is quirky...  Major ticks every 3 seconds, and then
# minor ticks every 0.75 s, are ugly.  The matplotlib alternatives aren't
# entirely satisfactory either.  MultipleLocator will put its major ticks
# at GPS timestamps that are themselves a multiple of the specified base -
# rather than at timestamps whose *distance* from the given epoch is a
# multiple of the base, which may result in the epoch not receiving a
# major tick when it isn't itself a multiple of the base.  (Sigh.)
# There's no way to please everyone:  Whatever we do here, the user may need
# to perform some mental arithmetic to work out timestamps of additional
# features in the plots, either absolute or relative to t0.
# Also, there's a tradeoff between marking tenths of a second and marking
# subdivisions of a second along powers of 2  (the latter are harder to
# read out loud, but more convenient for predicting what will be shown
# after zooming in).  Settling on a compromise;  AutoMinorLocator will
# take care of the details.
if t_width >= 32:
    t_epoch = 10 * floor(t0 / 10)
    t_major = 5
else:
    t_epoch = floor(t0)
    t_major = min(1, t_width / 8)

# One extra sample will ensure that the rightmost major tick will
# be drawn when t0 is a sufficiently round number (in binary)...
t_plotedge = t_plotend + 1./sample_rate

data_settings = AttributeHolder()
data_settings.t0 = t0
data_settings.t0_iso = t0_iso
data_settings.t_width = t_width
data_settings.t_plotstart = t_plotstart
data_settings.t_plotend = t_plotend
data_settings.t_plotedge = t_plotedge
data_settings.t_epoch = t_epoch
data_settings.t_major = t_major

# ---------------------------------------------------------------------------
# -- Data load processing...
# ---------------------------------------------------------------------------

st.info(f't0 = {t0} (GPS) = {t0_iso} (UTC)')

# The following calls for a single `st.caption()` call to avoid excessive
# vertical space, but also for hard line breaks via markdown.
override_acks = ''
if overrides.large_caches:
    override_acks += '''Using larger-sized strain data caches.\\
    '''
if overrides.wide_cache_blocks:
    override_acks += ('Using' if cache_wide_blocks else 'Allowing') \
        + ''' extra-wide cache blocks.\\
        '''
if overrides.url_caching:
    override_acks += '''GWpy URL cache is enabled.\\
    '''
if overrides.mem_profiling:
    override_acks += '''Memory profiling enabled;  watch the logs.\\
    '''
# Streamlit quirk:  `st.caption()` strips a trailing newline from its
# argument, thus a trailing backslash-newline would end up displaying
# a single backslash.  So we need to chop off this final pair ourselves.
# Unlike `str.removesuffix()`, which gets confused by the newline,
# the odd-looking `str.rstrip()` does the right thing.
if override_acks != '':
    st.caption(
        # pylint: disable-next=bad-str-strip-call
        override_acks.rstrip(
            '''\\
            '''
        )
    )

if floor(t_end / app_conf.CHUNK_SIZE) > floor(t_start / app_conf.CHUNK_SIZE):
    state_adv = \
        '''Brew a pot of :tea: while we're fetching some {0} strain
        data in {1} s chunks from GWOSC...'''
else:
    state_adv = \
        '''Grab a :coffee: while we're fetching a {1} s chunk of {0}
        strain data from GWOSC...'''
state_msg = state_adv.format(interferometer, app_conf.CHUNK_SIZE)
load_strain_state = st.markdown(state_msg)

try:
    strain, flag_data = load_strain(data_descriptor)
# pylint: disable-next=broad-exception-caught
except Exception:
    load_strain_state.markdown('')
    st.warning(
        f'''Load failed; data from {interferometer} may not be
        available on GWOSC for time {t0}, or the GWOSC
        data service might be temporarily unavailable.
        Please try a different time and/or interferometer.'''
    )
    emit_footer()
    st.stop()

loaded_msg = \
    f'Loaded {interferometer} strain data ({sample_rate} samples/s).'
load_strain_state.markdown(loaded_msg)
st.write('Cache block start:', t_start, ', end:', t_end,
         '; plot start:', t_plotstart, ', end:', t_plotend)

strain_cropped = strain.crop(
    t_plotstart,
    t_plotedge
)

# When the one extra sample at the edge would fall into the next data gap,
# we'll need to retreat.  (The subtle tell-tale will be a missing tick.)
if np.isnan(strain_cropped.value[-1]):
    strain_cropped = strain.crop(
        t_plotstart,
        t_plotend
    )

data = AttributeHolder()
data.strain = strain
data.strain_cropped = strain_cropped
data.flag_data = flag_data

# Explicit garbage collections may well be overkill, but if we want
# to do them at all, now is a good time - chances are we have just
# purged an old strain entry from the cache, and local variables
# from earlier completed runs have gone out of scope.
gc.collect()

# Summarize memory profiling if requested:
if overrides.mem_profiling:
    print('[-- After GC, before plotting --]')
    print(gc.get_stats())
    print_mem_profile(6)

# ---------------------------------------------------------------------------
# ...available data segments plot...
# ---------------------------------------------------------------------------

st.markdown('##### Data available for use in the loaded block:')

available_plotter.plot_available_data_segments(
    data,
    data_descriptor,
    data_settings
)

# ---------------------------------------------------------------------------
# ...and raw data plot --
# ---------------------------------------------------------------------------

st.subheader('Raw strain data')

# This might find no data to plot when the requested interval falls inside
# a data gap.  The plot() method would fail silently without raising an
# exception;  only the funny tick labels would leave the viewer scratching
# their head.  But there's a slightly obscure tell-tale:

try:
    if np.isnan(strain_cropped.max().to_value()):
        raise DataGapError()

    with _lock:
        figure_raw = strain_cropped.plot(color=appearance.PRIMARY_COLOR)

        raw_title = f'{interferometer}, around {t0} ({t0_iso} UTC), raw'
        ax = figure_raw.gca()
        ax.set_title(
            raw_title,
            loc='right',
            fontsize=appearance.RAW_TITLE_FONTSIZE
        )
        ax.set_xscale('seconds', epoch=t_epoch)
        if t_width >= 1.0:
            ax.xaxis.set_major_locator(MultipleLocator(base=t_major))
        if t_width <= 4.0:
            ax.xaxis.set_minor_locator(AutoMinorLocator(n=5))
        ax.set_ylabel('dimensionless')
        if raw_vline_enabled:
            ax.axvline(t0, color=appearance.VLINE_COLOR, linestyle='--')
        st.pyplot(figure_raw, clear_figure=True)

except DataGapError:
    st.error('t0 is too close to or inside a data gap. Please try a shorter'
             ' time interval, or try changing the requested timestamp.')

    emit_footer()
    st.stop()

st.divider()

# ---------------------------------------------------------------------------
# -- Filtered data plot processing --
# ---------------------------------------------------------------------------

if do_plot:
    st.subheader('Filtered data')

    # This will bail out with an error message when confronted with a
    # requested frequency range whose lower and upper bounds coincide.
    filtered_plotter.plot_filtered_data(
        data,
        data_descriptor,
        data_settings
    )

else:
    st.write('(Skipping filtered data plotting.)')

st.divider()

# ---------------------------------------------------------------------------
# -- ASD spectrum processing --
# ---------------------------------------------------------------------------

if do_show_asd:
    st.subheader('Amplitude spectral density')

    # Unlike bandpass filter construction, this handles a zero-width
    # frequency range gracefully by automatically expanding the range
    # (to a whole decade!).  Leaving this in as an Easter egg;  it can
    # be (ab)used to look beyond the upper frequency cutoff.
    asd_plotter.plot_asd_spectrum(
        data,
        data_descriptor,
        data_settings
    )

else:
    st.write('(Skipping ASD spectrum plot.)')

st.divider()

# ---------------------------------------------------------------------------
# -- Spectrogram processing --
# ---------------------------------------------------------------------------

if do_spec:
    st.subheader('Spectrogram')

    # Unlike other visualizations, spectrograms overlapping a data gap
    # would fail gracefully, leaving that part blank but keeping the
    # time axis and all the ticks where we want them to be - and unlike
    # others, they never depend on data outside the plotted time interval,
    # so there's no "too close to a data gap" case.
    # But we never get here when the plotted interval overlaps a data gap
    # by more than its endpoint, since we have the raw data plot section
    # bail out and stop the script in this case (for consistency).

    spec_plotter.plot_spectrogram(
        data,
        data_descriptor,
        data_settings
    )
else:
    st.write('(Skipping spectrogram.)')

st.divider()

# ---------------------------------------------------------------------------
# -- Q-transform processing --
# ---------------------------------------------------------------------------

if do_qtsf:
    st.subheader('Constant-Q transform')

    qtsf_plotter.plot_q_transform(
        data,
        data_descriptor,
        data_settings
    )
else:
    st.write('(Skipping Q-transform rendering.)')

# ---------------------------------------------------------------------------
# -- End game --
# ---------------------------------------------------------------------------

emit_footer()

# Summarize memory profiling if requested:
if overrides.mem_profiling:
    print('[-- At script end --]')
    print_mem_profile(8)

# ---------------------------------------------------------------------------
# A few Pylint notes:
# 1. Pylint (running as a GitHub workflow) only knows about the standard
# library.  It has no way of looking inside matplotlib.ticker to see that
# our custom Formatter needs no further public methods, nor of looking inside
# Streamlit to see that its cache_data decorators really use our "unused"
# additional function arguments.
# 2. It does not know that this script, as a web application, needs to be
# the exception handler of last resort.  Yes, we do want to catch *all*
# kinds of exceptions in several places and display a meaningful message
# rather than a Python backtrace to the end user when there's nothing
# better we can do.
# 3. It has some ideas of its own about what is a "constant" and what is a
# "variable", and they do not always agree with how identifiers are used in
# a typical Streamlit setting.  I'm not entirely convinced that following
# its uppercase recommendations improves readability in all cases.
# Then again, it *doesn't* warn about identifiers that *are* used to
# refer to constant non-scalar things, like lists or dicts...
# 4. Then again, it is perfectly happy with our mixing of predefined
# constants with ad-hoc one-shot literal constants.  Different choices
# could have been made in this regard - it is not always clear what's
# going to be best for readability.

# pylint: disable=too-many-lines
