## Installing the web app for local use[^1]

### (0) Prerequisites:

- A web browser (Firefox recommended[^2]),
- network connectivity to the [Gravitational Wave Open Science Center](https://gwosc.org/) website,
- a working Python 3.7 (or newer) runtime,
- [Pipenv](https://pipenv.pypa.io/en/latest/),
- at least 1 GiB of RAM to spare,
- about 450 MiB of disk space to hold a Python 3.9 virtual environment,
- `git` (optional).

### (1) Fetch the code:

#### If you just want to be able to run it (without making changes to the code):

You can download the [repository](https://github.com/GNiklasch/GWO-glitch-visualization/) as a ZIP archive (from the green `<> Code` button's menu) and unpack it under a convenient parent directory.

#### If you want to experiment with modifications:

You should have a working `git` utility and be familiar with its basic functionality. Either clone the original repository:

```
git clone https://github.com/GNiklasch/GWO-glitch-visualization.git
```
or, if you already have a [GitHub](https://github.com/) account, first create your own fork of the [original repository](https://github.com/GNiklasch/GWO-glitch-visualization/) on GitHub and then clone that to your local system.

### (2) Prepare the virtual environment:

#### (2.1) Change into the `GWO-glitch-visualization` directory
of the unpacked or cloned repository at your commandline prompt in a terminal window.

#### (2.2) Create the virtual environment:
```
pipenv install
```
This will automatically pull in [`GWpy`](https://gwpy.github.io) and [`Streamlit`](https://streamlit.io) and all their dependencies.

### (3) Launch the application:

#### (3.1) Still in the `GWO-glitch-visualization` directory, enter the virtual environment:
```
pipenv shell
```
You are now in a subshell which is aware of the virtual environment and knows about the executable `streamlit` wrapper.

#### (3.2) Start the web application:

```
streamlit run src/python/glitchplot.py
```
After a few moments, `streamlit` will ask your browser to open the web user interface on `http://localhost:8501/` (unless you have already made changes to the settings). After a few further moments, it should be ready to use!

#### (3.2.1) Commandline options:

If you have several GiB of RAM to spare, try
- `-C` to allow the application to **c**ache data from a larger number of different interferometer/timestamp combinations, and/or
- `-W` to have it cache **w**ider intervals of data (a few minutes) from a given interferometer around a given timestamp. There's also
- `-M` to monitor the **m**emory consumption. (This will produce quite a lot of somewhat technical terminal output to the terminal during each pass through the web app). As well as
- `-s` to **s**ilence the smallprint about cookies and the cloud hosting provider which isn't relevant when you use the application locally.

Options can be combined, and need to be separated from options to `streamlit` itself by a double hyphen. For example, I might use:
```
streamlit run src/python/glitchplot.py -- -CMWs
```
to apply all four of the above.

#### (3.2.2) Live code changes:

The running `streamlit` will notice it if you edit and save any of the Python source files. Click "Rerun" in the web UI to see the changed behavior (or any error messages about newly introduced bugs...).

### (4) To stop the application:
Interrupt the `streamlit` process in the terminal window (`ctrl-c` on UNIX/Linux/macOS systems), and close the now-disconnected browser window or browser tab.

To restart it later, return to step (3.2).

If you no longer need the pipenv shell, `exit` from it. You'll have to pick up at step (3.1) the next time.

### (5) *Enjoy!*

---

#### Footnotes:

[^1]: Or simply visit the web app
  [in the cloud](https://gwo-glitch-plotter.streamlit.app/).
  Tradeoffs: The data loads from GWOSC may run quicker in the cloud server,
  but it runs in a limited amount of RAM and therefore with a data cache
  limited in size, and you'll be sharing the cache with other users.
  Also, the cloud server may be rebooted at unexpected moments,
  disrupting all user sessions. Also, occasionally the whole Streamlit
  Community Cloud will be in maintenance.
  A local installation benefits from lots of RAM and from a fast network
  connection, and it allows you to experiment with code modifications.
[^2]: Safari has some peculiarities that don't play well with Streamlit-based
  web applications. In particular, it sometimes spontaneously decides to
  forget everything about the current page and to reload it from scratch,
  losing the current session state.
