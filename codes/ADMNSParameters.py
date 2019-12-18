import numpy as np
#########################################################
##### FUNDAMENTAL CONSTANTS (FROM PDG BOOKLET 2016) #####
##### ***** NEEDED TO CONVERT RESULTS IN SI ****** ######
#########################################################
c=299792458.
c2,c3,c4,c5=c**2,c**3,c**4,c**5
G=6.67408e-11
M0=1.98848e30
h=6.626070040e-34
e=1.6021766208*1e-19
e2=e**2.
mu0=4.*np.pi*1e-7
eps0=1./(mu0*c2)
hbar=h/(2*np.pi)
hbar2=hbar**2
alpha=e2/(4.*np.pi*eps0*hbar*c)
me=0.5109989461*1e6 # electron mass in eV
mekg=me*e/c2
ebis=np.sqrt(4*np.pi*alpha)  # electric charge in natural units
Gbis=G*e2/(hbar*c5) # Gravitational constant in natural units
#####################
### NATURAL UNITS ### (conversion factors SI - natural units)
#####################
tnu=e/hbar
dnu=e/(hbar*c)
mnu=c2/e
vnu=1./c
Pnu=(hbar/e2)

"""
Two sets of parameters describing the neutron star and the 
axion dark matter

"""

#############################################################################
### SET N 1: PARAMETERS FOR TYPICAL ALIGNED CONFIGURATION
#############################################################################
##########################
#### INPUT IN SI UNITS ###
##########################
M=1*M0           #[kg]
rns=1e4          #[m]
vai=250*1e3      #[m/s]
vdisp=200*1e3    #[m/s]
rho=0.3*1e9*1e6  #[eV/m^3]
g=1e-21          #[ev^-2]
tns=0            #[s]
##########################
ma=1e9*hbar/e  #[eV]
B0=1e14        #[G]
P=1            #[s]
thetam=0       #[rad]
thetai=np.pi/2 #[rad]
phii=0         #[rad]
#######################################
#### CONVERT INPUT IN NATURAL UNITS ### (electron Volts here !)
######################################
#### NS MASS IN ONLY USED FOR SCHWAR.
#### TRAJECTORIES 
#######################################
M=M*mnu
rns=rns*dnu
vai=vai*vnu
vdisp=vdisp*vnu
rho=rho/(dnu**3)
P=P*tnu
B0=B0*1.953*1e-2
tns=tns/tnu
#######################################
#######################################
params1={"thetai":thetai,"phii":phii,"M":M,"rns":rns,"ma":ma,"g":g,"B0":B0,"P":P,"thetam":thetam,"vai":vai,"tns":tns,"vdisp":vdisp,"rho":rho}


#############################################################################
### SET N 2: PARAMETERS OF FIGURE (S2) BEN
#############################################################################
##########################
#### INPUT IN SI UNITS ###
##########################
M=1*M0           #[kg]
rns=1e4          #[m]
vai=250*1e3      #[m/s]
vdisp=200*1e3    #[m/s]
rho=0.3*1e9*1e6  #[eV/m^3]
g=1e-21          #[ev^-2]
tns=0            #[s]
##########################
##########################
ma=5*1e-7       #[eV]
B0=2.5e13       #[G]
P=11.37         #[s]
thetam=15*(np.pi/180) #[rad]
thetai=58.31007808870454*(np.pi/180) #[rad]
phii=0 #[rad]
#######################################
#### CONVERT INPUT IN NATURAL UNITS ###
#######################################
M=M*mnu
rns=rns*dnu
vai=vai*vnu
vdisp=vdisp*vnu
rho=rho/(dnu**3)
P=P*tnu
B0=B0*1.953*1e-2
tns=tns/tnu
#######################################
#######################################
params2={"thetai":thetai,"phii":phii,"M":M,"rns":rns,"ma":ma,"g":g,"B0":B0,"P":P,"thetam":thetam,"vai":vai,"tns":tns,"vdisp":vdisp,"rho":rho}



