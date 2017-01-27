# openquake-inslab-builder
The OpenQuake Inslab Model Builder contains tools for constructing subduction inslab models for use with the OpenQuake-engine, and Jupyter Notebooks illustrating their application

The tools currently implement the novel method presented in Weatherill <sup>1</sup>, Pagani and Garcia (2017), but further code will be added to incorporate all of the
different methodologies used for the PEER Test comparison described therein.

Dependencies:
* [OpenQuake-engine](https://github.com/gem/oq-engine)
* [Matplotlib](http://matplotlib.org/)
* [Shapely](https://pypi.python.org/pypi/Shapely) - included in OpenQuake
* Jupyter Notebook (for demos)


To-do:

* Add 'fill-with-faults' methodology

* Add 'area source staircase' methodology

* Add Notebooks for all PEER Test Examples

Sample data is taken from [Slab 1.0](https://earthquake.usgs.gov/data/slab/)<sup>2</sup>

<sup>1</sup>Weatherill, G., Pagani, M and Garcia, J. (2017) "Modelling In-slab Subduction
Earthquakes in PSHA: Current Practice and Challenges for the Future" in 
Proceedings of the 16th World Conference on Earthquake Engineering, 16WCEE,
Santiago, Chile, January 9th to 13th 2017

<sup>2</sup>Hayes, G. P., D. J. Wald, and R. L. Johnson (2012), Slab1.0: A three-dimensional model of global subduction zone geometries, J. Geophys. Res., 117, B01302, doi:10.1029/2011JB008524. 
