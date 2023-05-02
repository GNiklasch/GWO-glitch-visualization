"""An interactive utility for plotting raw strain data from GWOSC,
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

# ---------------------------------------------------------------------------
# -- Imports and matplotlib backend sanitizing --
# ---------------------------------------------------------------------------

from math import ceil, floor
import argparse

# memory management and profiling
import gc
import tracemalloc

import streamlit as st
import numpy as np
from gwpy.timeseries import TimeSeries, StateTimeSeries
import astropy.time as atime
import matplotlib as mpl
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.ticker import LogFormatter, NullFormatter, \
    AutoMinorLocator, MultipleLocator, NullLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

# customizations
from plotutil.ticker import MyFormatter

# Importing customcm is required since it registers our custom colormap
# and its reversed form with matplotlib.
# pylint: disable=unused-import
import plotutil.customcm

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
overrides = parser.parse_args()

# (An alternative would have been to (ab)use .streamlit/secrets.toml to
# hold cloud-specific configuration settings.  They could then be tuned
# on the fly without a GitHub commit and redeployment.  But we'd still
# need *some* usable defaults here to enable people to run the code locally
# without creating a secrets.toml file, so this would not gain much.)

# ---------------------------------------------------------------------------
# -- Custom exceptions --
# ---------------------------------------------------------------------------

class ZeroFrequencyRangeError(ZeroDivisionError):
    """Runtime exception raised upon detecting a frequency range
    whose lower and upper limits coincide, which is unsuitable for
    constructing a band pass filter.
    """
    # pylint: disable=W0107
    pass

class DataGapError(ValueError):
    """Runtime exception raised to communicate that a gap in the
    available strain data prevents further processing and plotting.
    """
    # pylint: disable=W0107
    pass

# ---------------------------------------------------------------------------
# -- Helper methods: cacheable data...
# ---------------------------------------------------------------------------

# pylint: disable=W0621
def _load_strain_impl(interferometer, t_start, t_end, sample_rate=4096):
    """Workhorse wrapper around TimeSeries.fetch_open_data()"""
    # Work around bug #1612 in GWpy:  fetch_open_data() would fail if t_end
    # falls on  (or a fraction of a second before)  the boundary between
    # two successive 4096 s chunks.  Asking for a fraction of a second
    # *more* just past this boundary avoids the issue.
    t_end_fudged = t_end + 1/64.
    # (Is GWpy's URL caching thread-safe?  I certainly don't dare to turn
    # it on for multi-user operation in the cloud, without having control
    # over cache entry lifetimes.)
    # The other question is whether it's actually useful here...
    strain = TimeSeries.fetch_open_data(interferometer, t_start, t_end_fudged,
                                        sample_rate=sample_rate,
                                        cache=overrides.url_caching)
    intervals = int((t_end - t_start) * 8) - 1
    flag = StateTimeSeries(
        [not np.isnan(strain.value_at(t_start + i/8. + 1/16.))
         for i in range(intervals)],
        sample_rate=8,
        epoch=t_start).to_dqflag(round=False)
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
# pylint: disable=W0621
def load_low_rate_strain(interferometer, t_start, t_end, sample_rate=4096):
    """Cacheable wrapper around low-sample-rate data fetching"""
    return _load_strain_impl(interferometer, t_start, t_end,
                            sample_rate=sample_rate)

@st.cache_data(max_entries=8 if overrides.large_caches else 3)
# pylint: disable=W0621
def load_high_rate_strain(interferometer, t_start, t_end, sample_rate=16384):
    """Cacheable wrapper around high-sample-rate data fetching"""
    return _load_strain_impl(interferometer, t_start, t_end,
                            sample_rate=sample_rate)

# Recomputing a spectrogram or a const-Q transform doesn't take as long
# as plotting the results does, but caching them may still improve the
# user experience a little.
# Streamlit can't use the strain as the hash key, but it *can* use what
# we had used to fetch it - whence the dummy arguments.
@st.cache_data(max_entries=10 if overrides.large_caches else 4)
# pylint: disable=W0621
def make_specgram(_strain, interferometer, t_start, t_end, sample_rate,
                  t_plotstart, t_plotend, stride, overlap):
    """Cacheable wrapper around TimeSeries.spectogram()"""
    # pylint: disable=unused-argument
    specgram = _strain.spectrogram(stride=stride, overlap=overlap) ** (1/2.)
    return specgram

@st.cache_data(max_entries=16 if overrides.large_caches else 4)
# pylint: disable=W0621
def transform_strain(_strain, interferometer, t_start, t_end, sample_rate,
                     t_plotstart, t_plotend, t_pad, q, whiten):
    """Cacheable wrapper around TimeSeries.q_transform(), with graceful
    backoff to reduced padding when we're (too) close to a data gap"""
    outseg = (t_plotstart, t_plotend)
    # pylint: disable=unused-argument
    # Without nailing down logf and fres, q_transform() would default to a
    # very high value for the number of frequency steps, somehow resulting
    # in exorbitant memory consumption for the ad-hoc modified colormaps
    # created during plotting  (on the order of 380 MiB for a single
    # Q-transform plot at high sample rate!).
    fres = ceil(max(600, 24 * q) * (1 if sample_rate < 16384 else 1.3))
    q_warning = 0
    try:
        # The q_transform output would be distorted when the available strain
        # segment is much longer and isn't symmetric around t0:  Above a
        # certain frequency  (which depends on the padding and on the Q-value)
        # all features would be displaced to the left.  Pre-cropping the data
        # prevents this  (and speeds up processing, too).  Pre-cropping too
        # tightly, however, would result in broader whitening artefacts and
        # potentially losing output at low frequencies.
        padding = min(2.5 * t_pad,
                      t_end - t_plotend,
                      t_plotstart - t_start)
        strain_cropped = _strain.crop(t_plotstart - padding,
                                      t_plotend + padding)
        q_gram = strain_cropped.q_transform(outseg=outseg, qrange=(q, q),
                                            logf = True, fres = fres,
                                            whiten=whiten, fduration=t_pad)
    except ValueError:
        q_warning = 1
        try:
            # ...with less padding:
            strain_cropped = _strain.crop(t_plotstart - t_pad,
                                          t_plotend + t_pad)
            q_gram = strain_cropped.q_transform(outseg=outseg, qrange=(q, q),
                                                logf = True, fres = fres,
                                                whiten=whiten,
                                                fduration=t_pad)
        except ValueError:
            q_warning = 2
            # One last try, with no padding:
            strain_cropped = _strain.crop(t_plotstart, t_plotend)
            # Here, the default fduration=2 applies.
            q_gram = strain_cropped.q_transform(outseg=outseg, qrange=(q, q),
                                                logf = True, fres = fres,
                                                whiten=whiten)
            # If this last-ditch attempt fails, the exception is raised up
            # to our call site.
    return (q_gram, q_warning)

# ---------------------------------------------------------------------------
# ...timestamp conversion...
# ---------------------------------------------------------------------------

# Unlike `datetime`, `astropy` treats leap seconds correctly.

def gps_to_isot(val):
    """Convert a GPS timestamp to a UTC date/time in ISO 8601 format with
    a literal 'T' separating date and time."""
    # pylint: disable=redefined-builtin
    return atime.Time(val=atime.Time(val=val, scale='tai', format='gps'),
                      scale='utc', format='isot').to_string()

def iso_to_gps(val, format='isot'):
    """Convert a UTC date/time in ISO 8601 format with a literal 'T'
    separating date and time to a GPS timestamp."""
    # pylint: disable=redefined-builtin
    return atime.Time(val=atime.Time(val=val, scale='utc', format=format),
                      scale='tai', format='gps').to_value('gps')

def any_to_gps(val):
    """Convert the user intput to a GPS timestamp, accepting either
    UTC formatted as ISO 8601 date/time with 'T' or space separating
    time and date, optionally with a trailing 'Z', or text that can be
    parsed as a floating point number representing a GPS timestamp."""
    try:
        t = iso_to_gps(val=val)
    # pylint: disable=broad-exception-caught
    except Exception:
        try:
            # pylint: disable=redefined-builtin
            t = iso_to_gps(val=val, format='iso')
        # pylint: disable=broad-exception-caught
        except Exception:
            t=float(val)
    return t

# ---------------------------------------------------------------------------
# ...memory profiling...
# ---------------------------------------------------------------------------

def print_mem_profile(tops = 8) -> None:
    """Print out some memory diagnostics."""
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')
    print('[- Top {0} -]'.format(tops))
    for stat in top_stats[:tops]:
        print(stat)
    print('----------------')

# ---------------------------------------------------------------------------
# ...page footer --
# ---------------------------------------------------------------------------

def emit_footer() -> None:
    """Emit the page footer."""
    st.divider()
    stamp = atime.Time(atime.Time.now(),
                       scale='utc', format='isot').to_string()
    FOOTER = r"""
    View the [source code on GitHub]({0}).\\
    Inspired by [GW Quickview]({1}).\\
    Powered by [GWpy]({2}); fed with [data]({3}) hosted by the [GWOSC]({4}).\\
    Web user interface created with [Streamlit]({5}).\\
    \- Page refreshed {6} UTC.
    """.format('https://github.com/GNiklasch/GWO-glitch-visualization',
               'https://github.com/jkanner/streamlit-dataview/',
               'https://gwpy.github.io/',
               'https://gwosc.org/data/',
               'https://gwosc.org/',
               'https://streamlit.io/',
               stamp)
    st.markdown(FOOTER)

# ---------------------------------------------------------------------------
# -- Appearance:  Page layout...
# ---------------------------------------------------------------------------

APPTITLE = 'GWO glitch plotter'
st.set_page_config(page_title=APPTITLE, page_icon=":sparkler:")

# CSS tweaks:  Hide Streamlit's own version of the footer, and get rid
# of excessive vertical padding while we're at it.
# This must be emitted right at the start  (before emitting the title!)
# in order to have the page look right immediately when it is visited.
HIDE_ST_FOOTER = """
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
"""
st.markdown(HIDE_ST_FOOTER, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# ...colors and colormaps...
# ---------------------------------------------------------------------------

PRIMARY_COLOR = st.get_option("theme.primaryColor") # '#0F2CA4'
VLINE_COLOR = 'orange'

# Both of the following are derived from PRIMARY_COLOR and will produce
# very nearly the same shade when plotted on a white background.  The
# transparent version becomes invisible when plotted over the PRIMARY_COLOR,
# making it possible to plot a background curve after plotting the foreground
# curve.
# (If one wants the background curve to be faintly visible "through" the
# foreground one, the transparent color shade could be made slightly lighter
# and slightly more opaque, interpolating between the current two specs.)
ASD_LIGHT_COLOR = '#7A89C8'
ASD_TRANSPARENT_COLOR = PRIMARY_COLOR + '96' # '#0F2CA496'

colormaps = {'Viridis (Gravity Spy)': 'viridis',
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
colormap_choices = list(colormaps)

# ---------------------------------------------------------------------------
# ...font sizes...
# ---------------------------------------------------------------------------

RAW_TITLE_FONTSIZE = 14
FILTERED_TITLE_FONTSIZE = 14
ASD_TITLE_FONTSIZE = 10
ASD_LABEL_FONTSIZE = 11
ASD_LABEL_LABELSIZE = 10
SPEC_TITLE_FONTSIZE = 17
QTSF_TITLE_FONTSIZE = 17

# ---------------------------------------------------------------------------
# ...and the (in-page) title:
# ---------------------------------------------------------------------------

st.title('Plot glitches from GWOSC-sourced strain data')

# ---------------------------------------------------------------------------
# -- Input selectables and related parameters --
# ---------------------------------------------------------------------------

# Strictly speaking the following is valid for O3 only...
calib_freqs_low = {'H1': 10, 'L1': 10, 'V1': 20}
interferometers = list(calib_freqs_low)

# The L1 view of GW170817, with the almighty ETMY saturation / ESD overflow
# glitch a second earlier, is well known as the Gravity Spy logo.
INITIAL_T0_GPS = '1187008882.4'

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
T_ELBOW_ROOM = 46.7

# Filtering will have to get by with less padding beyond the half width:
T_PAD = 8.0

t_widths = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8, 16, 32, 64]
INITIAL_WIDTH = t_widths[5]

sample_rates = [4096, 16384]

CHUNK_SIZE = 4096 # only used for informative messages

# It might be nice to have a red "Don't...." and green "Do...", but Streamlit
# selectbox options do not support markdown/colors.  Moreover, selecting any
# particular selectbox widget for CSS manipulations is awkward at best, and
# having the CSS settings depend on the textual content of the <div> element
# is not practical.
# (Radio button options don't support markdown either;  and radio buttons
# would take up too much space, as well as breaking the flow of the sentence.)
yorn = {"Don't...": False, 'Do...': True}
yorn_choices = list(yorn)

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
f_detents = [8, 9.51, 11.3, 13.5, 16, 19.0, 22.6, 26.9,
             32, 38.3, 45.3, 53.8, 64, 76.1, 90.5, 108,
             128, 152, 181, 215, 256, 304, 362, 431,
             512, 609, 724, 861, 1024, 1218, 1448, 1722,
             2048, 2435, 2896, 3444, 4096, 4871, 5793]

# ASD spectra and spectrograms share some of their input choices.
asd_y_decades = {-27: 1.E-27, -26: 1.E-26, -25: 1.E-25, -24: 1.E-24,
                 -23: 1.E-23, -22: 1.E-22, -21: 1.E-21, -20: 1.E-20,
                 -19: 1.E-19, -18: 1.E-18, -17: 1.E-17, -16: 1.E-16}
asd_y_detents = list(asd_y_decades)
asd_offsets = {'-12': -12, '-6': -6, '-3': -3, '-1.5': -1.5,
               'None': 0,
               '1.5': 1.5, '3': 3, '6': 6, '12': 12}
asd_offset_detents = list(asd_offsets)
ASD_INITIAL_OFFSET = asd_offset_detents[4] # 'None'
asd_initial_y = (asd_y_detents[1], asd_y_detents[-4])
spec_initial_v = (asd_y_detents[1], asd_y_detents[-4])

# Spectrograms will have 8 Hz resolution except for very short widths where
# the stride will have to be adjusted down.  4 Hz (0.25 s stride) is too fine;
# the LIGO fundamental violin modes would then get lost under the horizontal
# grid lines.
spec_stride = 0.125

# View limit frequencies for spectrograms:
spec_f_detents = [10, 22.6, 45.3, 90.5, 181, 362, 724, 1448, 2896, 5793]

# Q-values spaced at powers of sqrt(2):
q_values = [5.66, 8, 11.3, 16, 22.6, 32, 45.3, 64]
INITIAL_Q = q_values[2]

# Colormap scaling for const-Q transforms:
normalized_energies = [6.3, 12.7, 25.5, 51.1, 102.3]
INITIAL_NE_CUTOFF = normalized_energies[2]

# Start memory profiling if requested:
if overrides.mem_profiling:
    tracemalloc.start()

# ---------------------------------------------------------------------------
# -- Command and control sidebar --
# ---------------------------------------------------------------------------

st.sidebar.markdown('# Controls')

st.sidebar.write('In each panel below, make adjustments as desired,'
                 ' then click the button underneath to apply them.')

# ... data-loading form:
with st.sidebar.form('load_what'):
    st.markdown('### Load what from where:')

    interferometer = st.selectbox('**From interferometer:**',
                                  interferometers, index=1) # default L1
    t_width = st.select_slider('**load enough to visualize**',
                               t_widths, value=INITIAL_WIDTH)
    t0_text = st.text_input('**seconds of strain data around GPS'
                            ' (or ISO 8601-formatted UTC) timestamp:**',
                            INITIAL_T0_GPS)
    cache_wide_blocks = False
    if overrides.wide_cache_blocks:
        cache_wide_blocks = st.checkbox(r'\- use wide cache blocks',
                                        value=True)
    sample_rate = st.selectbox('**Sample rate:**', sample_rates)

    load_submitted = st.form_submit_button('Load and plot raw data',
                                           type='primary',
                                           use_container_width=True)

# Preprocess parameters which depend on the sample rate and/or interferometer:
if sample_rate < 16384:
    f_detents_eff = f_detents[0:30]
    asd_f_detents_eff = f_detents[0:31]
    spec_f_detents_eff = spec_f_detents[0:8]
    spec_figsize = (12, 6)
    qtsf_figsize = (12, 7)
    load_strain = load_low_rate_strain
else:
    f_detents_eff = f_detents[0:38]
    asd_f_detents_eff = f_detents
    spec_f_detents_eff = spec_f_detents
    spec_figsize = (12, 7)
    qtsf_figsize = (12, 8)
    load_strain = load_high_rate_strain

f_initial_range = (f_detents_eff[1], f_detents_eff[28])
asd_initial_f_range = (asd_f_detents_eff[1], asd_f_detents_eff[-1])
spec_initial_f_range = (spec_f_detents_eff[0], spec_f_detents_eff[-1])

calib_freq_low = calib_freqs_low[interferometer]
calib_caveat = ("Caution: Strain data below {0} Hz from {1} aren't"
                ' calibrated.'
                ).format(calib_freq_low, interferometer)

# pylint: disable=implicit-str-concat
st.sidebar.caption((' Use the "Do..." and "' "Don't..." '" options to select'
                    ' which plots to show or skip.'))

# ... filtered-plot form:
with st.sidebar.form('plot_how'):
    # Default here is "Don't".
    do_plot_txt = st.selectbox('Shall we plot?', yorn_choices,
                               label_visibility = 'collapsed')
    do_plot = yorn[do_plot_txt]

    st.markdown('### ...filter and plot filtered data:')

    f_range = st.select_slider('**Bandpass limits [Hz]:**',
                               f_detents_eff, value=f_initial_range)
    whiten_plot = st.checkbox(r'\- whiten before filtering', value=True)
    filtered_vline_enabled = st.checkbox((r'\- highlight t0 in raw and'
                                          ' filtered data plots'),
                                         value=True)

    plot_submitted = st.form_submit_button('Apply filtered plot'
                                           ' settings',
                                           type='primary',
                                           use_container_width=True)

# ... ASD spectrum form:
with st.sidebar.form('asd_how'):
    # Default is "Don't".
    do_show_asd_txt = st.selectbox('Shall we show ASD?', yorn_choices,
                                   label_visibility = 'collapsed')
    do_show_asd = yorn[do_show_asd_txt]

    st.markdown('### ...show amplitude spectral density as a spectrum:')

    asd_f_range = st.select_slider('**Spectrum frequency range [Hz]:**',
                                   asd_f_detents_eff,
                                   value=asd_initial_f_range)
    asd_y_low, asd_y_high = st.select_slider(('**Spectrum ASD range,'
                                              ' decades:**'),
                                             asd_y_detents,
                                             value=asd_initial_y)
    asd_y_range = (asd_y_decades[asd_y_low], asd_y_decades[asd_y_high])
    asd_offset_choice = st.select_slider(('**Optional background ASD spectrum,'
                                          ' from [s] earlier or later:**'),
                                         asd_offset_detents,
                                         value=ASD_INITIAL_OFFSET)
    asd_offset = asd_offsets[asd_offset_choice]
    # Just in case someone wants to extract a light-shaded plot:
    asd_lighten = st.checkbox(r'\- swap shades: light foreground (and heavy'
                              ' background)')

    asd_submitted = st.form_submit_button('Apply spectrum settings',
                                          type='primary',
                                          use_container_width=True)

# ... Spectrogram form:
with st.sidebar.form('spec_how'):
    # Default here is "Don't".
    do_spec_txt = st.selectbox('Shall we spec?', yorn_choices,
                               label_visibility = 'collapsed')
    do_spec = yorn[do_spec_txt]

    st.markdown('### ...show a spectrogram:')
    spec_f_range = st.select_slider('**Spectrogram frequency range [Hz]:**',
                                   spec_f_detents_eff,
                                   value=spec_initial_f_range)
    spec_v_low, spec_v_high = st.select_slider(('**Spectrogram ASD range,'
                                                ' decades:**'),
                                               asd_y_detents,
                                               value=spec_initial_v)
    spec_v_min, spec_v_max = (asd_y_decades[spec_v_low],
                              asd_y_decades[spec_v_high])
    spec_grid_enabled = st.checkbox(r'\- enable grid overlay',
                                    value=True)
    spec_vline_enabled = st.checkbox(r'\- highlight t0',
                                     value=True)

    # Default is our custom Jetstream colormap, which is similar to one
    # used in the Virgo electronic logs.
    # Streamlit doesn't allow negative indices counting backward from
    # the end of the selectbox options...
    spec_colormap_choice = st.selectbox('**Spectrogram colormap:**',
                                        colormap_choices,
                                        index=len(colormap_choices)-2)
    spec_colormap = colormaps[spec_colormap_choice]

    spec_submitted = st.form_submit_button('Apply spectrogram settings',
                                           type='primary',
                                           use_container_width=True)

# ... Q-transform form:
with st.sidebar.form('qtsf_how'):
    # Default here is 'Do'.
    do_qtsf_txt = st.selectbox('Shall we qtsf?', yorn_choices,
                               label_visibility = 'collapsed',
                               index=1)
    do_qtsf = yorn[do_qtsf_txt]

    st.markdown('### ...render a constant-Q transform:')

    q0 = st.select_slider('**Q-value:**', q_values, value=INITIAL_Q)
    ne_cutoff = st.select_slider('**Normalized energy cutoff:**',
                                 normalized_energies,
                                 value=INITIAL_NE_CUTOFF)
    whiten_qtsf = st.checkbox(r'\- whiten before transforming', value=True)
    qtsf_grid_enabled = st.checkbox(r'\- enable grid overlay',
                                    value=True)
    qtsf_vline_enabled = st.checkbox(r'\- highlight t0',
                                     value=True)
    # Default is Viridis  (same as Gravity Spy's):
    qtsf_colormap_choice = st.selectbox('**Q transform colormap:**',
                                        colormap_choices,
                                        index=0)
    qtsf_colormap = colormaps[qtsf_colormap_choice]

    qtsf_submitted = st.form_submit_button('Apply Q transform settings',
                                           type='primary',
                                           use_container_width=True)

# ---------------------------------------------------------------------------
# -- Data load preparations --
# ---------------------------------------------------------------------------

try:
    # pylint: disable=bad-str-strip-call
    t0=any_to_gps(t0_text.strip())
except ValueError as ex:
    st.warning('Sorry, there seems to be a typo in the timestamp input:')
    st.error(ex)
    st.warning('Please correct and re-submit your load request.')
    emit_footer()
    st.stop()

t0_iso=gps_to_isot(t0)

t_cache_boundaries = ((512 if sample_rate < 16384 else 256)
                      if cache_wide_blocks else 32)

t_halfwidth = t_width / 2
t_start = t_cache_boundaries * floor((t0 - T_ELBOW_ROOM)/t_cache_boundaries)
t_end = t_cache_boundaries * ceil((t0 + T_ELBOW_ROOM)/t_cache_boundaries)
t_plotstart = t0 - t_halfwidth
t_plotend = t0 + t_halfwidth

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

# ---------------------------------------------------------------------------
# -- Data load processing...
# ---------------------------------------------------------------------------

st.info('t0 = {0} (GPS) = {1} (UTC)'.format(t0, t0_iso))

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
# Streamlit quirk:  st.caption() strips a trailing newline from its
# argument, thus a trailing backslash-backslash-newline would end up
# displaying a single backslash.
if override_acks != '':
    st.caption(override_acks.strip('''\\
    '''))

if floor(t_end / CHUNK_SIZE) > floor(t_start / CHUNK_SIZE):
    state_adv = ("Brew a pot of :tea: while we're fetching some {0} strain "
                 "data in {1} s chunks from GWOSC...")
else:
    state_adv = ("Grab a :coffee: while we're fetching a {1} s chunk of {0} "
                 "strain data from GWOSC...")
state_msg = state_adv.format(interferometer, CHUNK_SIZE)
load_strain_state = st.markdown(state_msg)

try:
    strain, flag_data = load_strain(interferometer,
                                    t_start, t_end, sample_rate)
# pylint: disable=broad-exception-caught
except Exception:
    load_strain_state.markdown('')
    st.warning(('Load failed; data from {0} may not be available on GWOSC for'
                ' time {1}, or the GWOSC data service might be temporarily'
                ' unavailable. Please try a different time and/or'
                ' interferometer.'
                ).format(interferometer, t0))
    emit_footer()
    st.stop()

loaded_msg = ('Loaded {0} strain data ({1} samples/s).'
              ).format(interferometer, sample_rate)
load_strain_state.markdown(loaded_msg)
st.write('Cache block start:', t_start, ', end:', t_end,
         '; plot start:', t_plotstart, ', end:', t_plotend)
strain_precropped = strain.crop(t_plotstart - T_PAD, t_plotend + T_PAD)

# One extra sample ensures that the rightmost major tick will be drawn when
# t0 is a sufficiently round number (in binary)...
t_plotedge = t_plotend + 1./sample_rate
strain_cropped = strain.crop(t_plotstart, t_plotedge)

# ... *except* when that one sample would fall into the next data gap.
if np.isnan(strain_cropped.value[-1]):
    strain_cropped = strain.crop(t_plotstart, t_plotend)
else:
    pass

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

st.subheader('Raw strain data')

# ---------------------------------------------------------------------------
# ...and raw data plot --
# ---------------------------------------------------------------------------

# This might find no data to plot when the requested interval falls inside
# a data gap.  The plot() method would fail silently without raising an
# exception;  only the funny tick labels would leave the viewer scratching
# their head.  But there's a slightly obscure tell-tale:

try:
    if np.isnan(strain_cropped.max().to_value()):
        raise DataGapError()

    with _lock:
        figure_raw = strain_cropped.plot(color=PRIMARY_COLOR)

        RAW_TITLE = ('{0}, around {1} ({2} UTC), raw'
                     ).format(interferometer, t0, t0_iso)
        ax = figure_raw.gca()
        ax.set_title(RAW_TITLE, fontsize=RAW_TITLE_FONTSIZE)
        ax.set_xscale('seconds', epoch=t_epoch)
        if t_width >= 1.0:
            ax.xaxis.set_major_locator(MultipleLocator(base=t_major))
        if t_width <= 4.0:
            ax.xaxis.set_minor_locator(AutoMinorLocator(n=5))
        ax.set_ylabel('dimensionless')
        if filtered_vline_enabled:
            ax.axvline(t0, color=VLINE_COLOR, linestyle='--')
        st.pyplot(figure_raw, clear_figure=True)

except DataGapError:
    st.error('t0 is too close to or inside a data gap. Please try a shorter'
             ' time interval, or try changing the requested timestamp.')

    # And provide some information about the available vs. unavailable data
    # in the vicinity, based on our own inspection of what we got from GWOSC
    # (rather than expending yet more time to fetch various metadata):
    with _lock:
        figure_flag = flag_data.plot(figsize=(12, 1))
        ax = figure_flag.gca()
        ax.set_title('Available / unavailable data vs. requested interval:',
                     fontsize=RAW_TITLE_FONTSIZE)
        ax.set_xscale('seconds', epoch=floor(t0))
        if t_end - t_start == 128:
            # Major ticks appear every 15 seconds, subdivide accordingly:
            ax.xaxis.set_minor_locator(AutoMinorLocator(n=3))
        else:
            # The default subdivisions into 5 or 4 are fine for wide cache
            # blocks and for 96 s blocks.
            pass
        ax.yaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_locator(NullLocator())
        # Always highlight t0 in *this* diagram:
        ax.axvline(t0, color=VLINE_COLOR, linestyle='--')
        ax.axvline(t_plotstart, color=PRIMARY_COLOR, linestyle='-.')
        ax.axvline(t_plotend, color=PRIMARY_COLOR, linestyle='-.')
        st.pyplot(figure_flag, clear_figure=True)

    emit_footer()
    st.stop()

st.divider()

# ---------------------------------------------------------------------------
# -- Filtered data plot processing --
# ---------------------------------------------------------------------------

if do_plot:
    st.subheader('Filtered data')
    try:
        if f_range[0] >= f_range[1]:
            # (Constructing the filter would raise a ValueError.)
            raise ZeroFrequencyRangeError()

        if whiten_plot:
            filtered = strain_precropped.whiten().bandpass(f_range[0],
                                                           f_range[1])
            wh_note = ', whitened'
        else:
            filtered = strain_precropped.bandpass(f_range[0], f_range[1])
            wh_note = ''

        filtered_cropped = filtered.crop(t_plotstart, t_plotedge)

        # Filtering will have failed when we are too close to a data gap,
        # and it fails silently - there's no exception we could catch.
        # But there's a tell-tale in the data:
        if np.isnan(filtered.max()):
            raise DataGapError()

        FILTERED_TITLE = ('{0}, around {1} ({2} UTC){3},'
                          ' band pass: {4} - {5} Hz'
                          ).format(interferometer, t0, t0_iso,
                                   wh_note, f_range[0], f_range[1])

        with _lock:
            figure_filtered = filtered_cropped.plot(color=PRIMARY_COLOR)
            ax = figure_filtered.gca()
            ax.set_title(FILTERED_TITLE,
                         loc='right', fontsize=FILTERED_TITLE_FONTSIZE)
            if whiten_plot:
                ax.set_ylabel('arbitrary units')
            else:
                ax.set_ylabel('dimensionless')
            ax.set_xscale('seconds', epoch=t_epoch)
            if t_width >= 1.0:
                ax.xaxis.set_major_locator(MultipleLocator(base=t_major))
            if t_width <= 4.0:
                ax.xaxis.set_minor_locator(AutoMinorLocator(n=5))
            if filtered_vline_enabled:
                ax.axvline(t0, color=VLINE_COLOR, linestyle='--')
            st.pyplot(figure_filtered, clear_figure=True)

        if f_range[0] < calib_freq_low:
            st.warning(calib_caveat)
        else:
            pass
    except DataGapError:
        st.error('t0 is too close to (or inside) a data gap, unable to'
                 ' filter the data. Please try a shorter time interval or'
                 ' try changing the requested timestamp.')
    except ZeroFrequencyRangeError:
        st.warning('Please make the frequency range wider.')

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

    asd_bgnd_warning = False
    try:
        # Computing the ASD will fail when the cropped interval overlaps a
        # data gap.  Again the tell-tale symptom is somewhat obscure:
        strain_asd = strain_cropped.asd()
        if np.isnan(strain_asd.max().to_value()):
            raise DataGapError()

        if not asd_offset == 0:
            strain_bgnd_asd = strain.crop(t_plotstart + asd_offset,
                                          t_plotend + asd_offset).asd()
            if np.isnan(strain_bgnd_asd.max().to_value()):
                asd_bgnd_warning = True
            else:
                if asd_offset > 0:
                    asd_bgnd_label = '{0} s later'.format(asd_offset)
                else:
                    asd_bgnd_label = '{0} s earlier'.format(-asd_offset)
        else:
            pass

        ASD_TITLE = ('{0}, during {1} s around {2} GPS ({3} UTC)'
                     ).format(interferometer, t_width, t0, t0_iso)
        ASD_XLABEL = ('Frequency [Hz], {0} - {1} Hz'
                      ).format(asd_f_range[0], asd_f_range[1])
        ASD_YLABEL = r'Strain ASD [${\mathrm{Hz}}^{-1/2}$]'

        with _lock:
            figure_asd = strain_asd.plot(color=
                                         ASD_LIGHT_COLOR if asd_lighten
                                         else PRIMARY_COLOR)
            ax = figure_asd.gca()
            # Now that we have the axes configured, plotting a non-existent
            # background FrequencySeries would just fail silently - but no
            # need to even try when we already know it wouldn't work.
            if not asd_offset == 0 and not asd_bgnd_warning:
                ax.plot(strain_bgnd_asd,
                        label=asd_bgnd_label,
                        color=PRIMARY_COLOR if asd_lighten
                        else ASD_TRANSPARENT_COLOR)
                # We'll let matplotlib pick the best corner for the legend.
                # GWpy's custom handler_map creates an example line segment
                # that's rather thick, but with handler_map=None to reinstate
                # matplotlib's defaults it would be too thin.
                ax.legend(fontsize=ASD_TITLE_FONTSIZE)
            ax.set_title(ASD_TITLE, fontsize=ASD_TITLE_FONTSIZE,
                         loc='right', pad=10.)
            ax.xaxis.set_major_formatter(LogFormatter(base=10))
            ax.xaxis.set_minor_formatter(MyFormatter(asd_f_range))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.set_xlim(asd_f_range)
            ax.set_ylim(asd_y_range)
            ax.set_ylabel(ASD_YLABEL, fontsize=ASD_LABEL_FONTSIZE)
            ax.set_xlabel(ASD_XLABEL, fontsize=ASD_LABEL_FONTSIZE)
            ax.xaxis.set_tick_params(which='major',
                                     labelsize=ASD_LABEL_FONTSIZE)
            ax.xaxis.set_tick_params(which='minor',
                                     labelsize=ASD_LABEL_LABELSIZE)
            ax.yaxis.set_tick_params(which='major',
                                     labelsize=ASD_LABEL_LABELSIZE)
            st.pyplot(figure_asd, clear_figure=True)

        if asd_f_range[0] < calib_freq_low:
            st.warning(calib_caveat)
        else:
            pass
        if asd_bgnd_warning:
            st.warning('t0 is too close to a data gap, unable to include'
                       ' a background spectrum. Try changing the time'
                       ' offset.')
        else:
            pass

    except DataGapError:
        st.error('t0 is too close to (or inside) a data gap, unable to'
                 ' extract a spectrum. Try a shorter time interval or try'
                 ' varying the requested timestamp.')

else:
    st.write('(Skipping ASD spectrum plot.)')

st.divider()

# ---------------------------------------------------------------------------
# -- Spectrogram processing --
# ---------------------------------------------------------------------------

if do_spec:
    st.subheader('Spectrogram')

    SPEC_TITLE = ('{0}, around {1} GPS ({2} UTC)'
                  ).format(interferometer, t0, t0_iso)
    if spec_stride > t_width / 8:
        spec_stride = t_width / 8
    spec_overlap = spec_stride / 4
    specgram = make_specgram(strain_cropped,
                             interferometer, t_start, t_end, sample_rate,
                             t_plotstart, t_plotend,
                             stride=spec_stride, overlap=spec_overlap)
    # Unlike other visualizations, spectrograms overlapping a data gap
    # would fail gracefully, leaving that part blank but keeping the time
    # axis and all the ticks where we want them to be - and unlike others,
    # they never depend on data outside the plotted time interval, so there's
    # no "too close to a data gap" case.
    # But we never get here when the plotted interval overlaps a data gap
    # by more than its endpoint, since we have the raw data plot section
    # bail out and stop the script in this case (for consistency).

    with _lock:
        figure_spec = specgram.plot(figsize=spec_figsize)
        ax = figure_spec.gca()
        cax = make_axes_locatable(ax).append_axes("right",
                                                  size="5%", pad="3%")
        figure_spec.colorbar(label=r'Strain ASD [${\mathrm{Hz}}^{-1/2}$]',
                             cax=cax, cmap=spec_colormap,
                             vmin=spec_v_min, vmax=spec_v_max,
                             norm='log')
        ax.set_title(SPEC_TITLE, fontsize=SPEC_TITLE_FONTSIZE)
        cax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(spec_grid_enabled)
        cax.grid(spec_grid_enabled)
        ax.set_xscale('seconds', epoch=t_epoch)
        if t_width >= 1.0:
            ax.xaxis.set_major_locator(MultipleLocator(base=t_major))
        if t_width <= 4.0:
            ax.xaxis.set_minor_locator(AutoMinorLocator(n=5))
        ax.set_yscale('log', base=2)
        ax.set_ylim(spec_f_range)
        if spec_vline_enabled:
            ax.axvline(t0, color=VLINE_COLOR, linestyle='--')
        st.pyplot(figure_spec, clear_figure=True)
else:
    st.write('(Skipping spectrogram.)')

st.divider()

# ---------------------------------------------------------------------------
# -- Q-transform processing --
# ---------------------------------------------------------------------------

if do_qtsf:
    st.subheader('Constant-Q transform')

    try:
        q_gram, q_warning = transform_strain(strain, interferometer,
                                             t_start, t_end,
                                             sample_rate,
                                             t_plotstart, t_plotend, T_PAD,
                                             q=q0,
                                             whiten=whiten_qtsf)
        q_error = False
    except ValueError:
        q_warning = 0
        q_error = True

    if q_error:
        st.error('t0 is too close to (or inside) a data gap, unable to'
                 ' compute the Q-transform. Try a shorter time interval or'
                 ' try varying the requested timestamp.')
    else:
        q_wh_note = ', whitened' if whiten_qtsf else ''
        QTSF_TITLE = ('{0}, around {1} ({2} UTC), Q={3}{4}'
                      ).format(interferometer, t0, t0_iso,
                               q0, q_wh_note)

        with _lock:
            figure_qgram = q_gram.plot(figsize=qtsf_figsize)
            ax = figure_qgram.gca()
            cax = make_axes_locatable(ax).append_axes("right",
                                                      size="5%", pad="3%")
            figure_qgram.colorbar(label="Normalized energy",
                                  cax=cax, cmap=qtsf_colormap,
                                  clim = [0, ne_cutoff])
            ax.set_title(QTSF_TITLE, fontsize=QTSF_TITLE_FONTSIZE)
            ax.title.set_position([.5, 1.05])
            ax.grid(qtsf_grid_enabled)
            cax.grid(qtsf_grid_enabled)
            ax.set_xscale('seconds', epoch=t_epoch)
            if t_width >= 1.0:
                ax.xaxis.set_major_locator(MultipleLocator(base=t_major))
            if t_width <= 4.0:
                ax.xaxis.set_minor_locator(AutoMinorLocator(n=5))
            ax.set_yscale('log', base=2)
            ax.set_ylim(bottom=10)
            if qtsf_vline_enabled:
                ax.axvline(t0, color=VLINE_COLOR, linestyle='--')
            st.pyplot(figure_qgram, clear_figure=True)

        if q_warning > 0:
            q_caveat = ('t0 is close to a data gap, thus the Q-transform'
                        ' could not look{0} beyond the edges of what has'
                        ' been plotted and areas near these edges may'
                        ' contain artefacts.'
                        ' Also, information about low frequencies may be'
                        ' insufficient to paint that region.'
                        ).format(' far' if q_warning == 1 else '')
            st.warning(q_caveat)
        else:
            pass

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
