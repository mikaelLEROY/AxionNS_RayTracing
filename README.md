# AxionNS

This repository contains Python 3 ray-tracing code to compute radio light curves resulting from the resonant conversion of Axion dark matter into photons within the magnetosphere of a neutron star. 

Photon trajectories are traced from the observer to the magnetosphere where a root finding algorithm identifies the regions of resonant conversion. Given the modeling of the axion dark matter distirbution and conversion probability, one can compute the photon flux emitted from these regions. The individual contributions from all the trajectories is then summed to obtain the radiated photon power per unit solid angle.

Here we assume an isotropic PSD for the axion dark matter and take the magnetosphere to be that of an oblique rotating dipole, with charge densities computed within the Goldreich-Julian model. The conversion rate axion -> photon is discussed in the paper.
