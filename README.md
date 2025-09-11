**Phase Graph Tutorial**  

A collection of educational notebooks about Extended Phase Graphs (EPGs):

- `examples.ipynb` : basic examples to plot phase graphs and compute EPGs
- `plot_epg_states.ipynb` : plot the full EPG state evolution
- `tse_signal_shaping.ipynb` : compute variable flip-angles for TSE signal shaping

The notebooks use code from the python file `phase_graph.py`, which should be in the same directory:

-  `pg_plot()` - a function to plot phase graphs
-  `basic_epg()` - a python class for basic EPG calculations

The latter is only used, if the `pyepg` class is **not** available. This is a python wrapper for a fast cpp implementation of EPGs, see [here](https://github.com/mrphysics-bonn/EPGpp). 
Both classes have the same functionality, but `pyepg` is approximately 1000 times faster than `basic_epg`. With `basic_epg` some example may take several minutes.

