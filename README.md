**Phase Graph Tutorial**  

A collection of educational notebooks about Extended Phase Graphs (EPGs):

- `phase_graphs.ipynb` : plotting phase graphs 
- `epg_examples.ipynb` : basic EPG examples
- `plot_epg_states.ipynb` : plot the EPG state evolution
- `tse_signal_shaping.ipynb` : variable flip-angles for TSE signal shaping
- `epgOptimization_esmrmb25.ipynb` : ESMRMB 2025 tutorial notebook

The notebooks imports functionality from `epg_code.py`, which should be in the same directory:

-  `pg_plot()` - a function to plot phase graphs
-  `basic_epg()` - a python class for basic EPG calculations

The latter is only used, if the `pyepg` class is **not** available. This is a python wrapper for a fast cpp implementation of EPGs, see [here](https://github.com/mrphysics-bonn/EPGpp). 
Both classes have the same functionality, but `pyepg` is approximately 1000 times faster than `basic_epg`. With `basic_epg` some example may take several minutes.

