# About GWO glitch visualization

This repository holds source code for a tool to visualize *glitches* (transient noise events of non-astrophysical origin) in published gravitational-strain data recorded by the [LIGO](https://www.ligo.caltech.edu/) and [Virgo](https://www.virgo-gw.eu/) gravitational-wave observatories.

## Explore

- Visit the live web application deployed on the Streamlit Community Cloud:
[![GWO Glitch Plotter](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://gwo-glitch-plotter.streamlit.app)
- You can also [install](/doc/INSTALL.md) and run the application locally.

## Audience

You may find this glitch plotter web app useful if you are an eager Citizen Engineer contributing to the [Gravity Spy](https://www.zooniverse.org/projects/zooniverse/gravity-spy/) project on [Zooniverse](https://www.zooniverse.org/), if you have already been using the [Gravity Spy tools](https://gravityspytools.ciera.northwestern.edu/) to access glitch metadata such as the event timestamp and peak frequency, and if you have already seen hundreds or thousands of Q-transform images and are keen to look at further details beyond the familiar four-second frames. *What did the raw waveform look like at that moment? How does it look after some filtering? What was the noise-versus-frequency distribution (spectrum)? How did it change over the course of several seconds (comparison spectra; spectrogram)? What about transforming the data with other Q-values than Gravity Spy's choice, and viewing up to 64 seconds of context? And what happened a minute earlier, what happened a minute later?*

If you are chiefly interested in gravitational-wave detections rather than glitches, you may be better served by @jkanner's [GW Quickview](https://github.com/jkanner/streamlit-dataview/) web app.

Or use both web apps side by side if you like - each can do a few things that the other does not do.

## How it works

- Powered by [`GWpy`](https://gwpy.github.io/),
- fed with [data](https://gwosc.org/data/) hosted by the Gravitational Wave
Open Science Center ([GWOSC](https://gwosc.org/)),
- web user interface created with [Streamlit](https://streamlit.io/).

<!-- ## License (to be added) -->

## Acknowledgements

Profound thanks to the glitch-hunting volunteer communities of the [Gravity Spy](https://www.zooniverse.org/projects/zooniverse/gravity-spy/) and [GwitchHunters](https://www.zooniverse.org/projects/reinforce/gwitchhunters) projects for countless exciting questions and discussions, and for the teams of researchers for making it all possible.

Kudos to @jkanner for creating [GW Quickview](https://github.com/jkanner/streamlit-dataview/), empowering non-programmers to turn reams of raw data into impressive and enlightening plots with a few mouse clicks - and proving that this can be done with a surprisingly modest amount of code behind the scenes. While I have made [quite different choices to accommodate our rather different use case scenario](doc/Rationale.md), I would never have started without this inspiration.

Very special thanks to Zooniverse contributors \@Coralbell and \@Nicm25 for [bringing GW Quickview to our attention](https://www.zooniverse.org/projects/zooniverse/gravity-spy/talk/330/2833389), for working many examples, and for encouraging further playing with `GWpy` and `matplotlib`, from which the present development has grown.
