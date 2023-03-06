# IA-KP1
Code to perform measurements and analysis of FS2 mocks.

The measure folder yields code that perform wg+ and wgg measurements.
The notebook show examples on how to use the code.
The codes takes as input a density and shape catalog, and take either one, or two random catalogues depending on the situation.
So far the following test have been performed :
- The jackknife code works in every mode ('auto',cross' or 'subsample').
- The d-deleted jackknife code hasn't been tested yet in the subsample and cross configuration.

The theory folder needs to be update with notebook examples, so far it takes in account:
- modelling of wgg and wg+ in real space, taking or not into account the radial distribution of the sources.
- RSD modelling of wgg
- minimization with iminuit.

What is missing :
- RSD for wg+
- non linear bias term for wg+.
- implementation of emcee.

Requirements : 
- treecorr : https://rmjarvis.github.io/TreeCorr/_build/html/index.html
- pycorr : https://py2pcf.readthedocs.io/en/latest/
- pyccl : https://ccl.readthedocs.io/en/latest/
- fast-pt : https://github.com/JoeMcEwen/FAST-PT
- iminuit : https://iminuit.readthedocs.io/en/stable/
- emcee : https://emcee.readthedocs.io/en/stable/
