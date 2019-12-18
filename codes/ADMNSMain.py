"""
MAIN FUNCTIONS COMPUTING THE RADIATED POWER PER UNIT SOLIDE ANGLE

The relevant parameters are:

Axion Dark matter parameters:
- axion mass ma

Neutron Star parameters: 
- period P
- magnetic field at the surface B0
- inclination of magnetic dipole with respect to rotation axis: thetam

Viewing angle for the observer:
- thetai
- phii

The remaining parameters are either contributing to the radiated power as multiplicative factor or do not vary in practice. 
These are:
- axion two-photon coupling constant g
- neutron star radius rns
- neutron star mass M
- axion dark matter density at infinity rhoDMinf
- axion dark matter velocity dispersion at infinity vDMdisp
- axion dark matter mean velocity at infinity
"""

import numpy as np
import matplotlib
from decimal import Decimal
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy
import time
from scipy import integrate
from scipy.integrate import quad
from scipy import interpolate
from .ADMNSParameters import *
from .ADMNSTools import *

#####################################################################################
#################### MODEL AXION DARK MATTER + NEUTRON STAR #########################   
#####################################################################################

#############################################################
### ********** LOCAL AXION DARK MATTER DENSITY **************
#############################################################
### HERE WE ASSUME AN ISOTROPIC GAUSSIAN PHASE-SPACE DENSITY 
### AT INFINITY AND USE LIOUVILLE'S THEOREM
#############################################################
def integrand(t,x):
    return np.exp(-(t/x)**2.)*np.exp(-2.*t)
def Integrale(x):
    try:
        I=integrate.quad(integrand,0.,np.inf,args=(x))[0]
    except:
        I=np.vectorize(quad)(integrand,0.,np.inf,args=x)[0]
    return I
def Density(r,M,rhoDMinf,vDMdisp): #[rhoDMinf]
    x=np.sqrt(2.*Gbis*M/r)/vDMdisp
    rhoDM=rhoDMinf*(2./np.sqrt(np.pi))*x
    #rhoDM=rhoDMinf*(2./np.sqrt(np.pi))*(x+(1./x)*Integrale(x))
    #try:
    #    rhoDM=rhoDMinf*(2./np.sqrt(np.pi))*(x+(1./x)*Integrale(x))
    #except:
    #    rhoDM=np.array([])
    return rhoDM    

################################################################
### ************** NEUTRON STAR MAGNETIC FIELD *****************
################################################################
### HERE WE USE THE MAGNETIC FIELD OF AN OBLIQUE ROTATING DIPOLE
################################################################
def MagneticField(r,theta,phi,B0,P,thetam,tns,rns): #[Gauss]
    Omega=(2.*np.pi)/P
    MB=0.5*B0*((rns/r)**3.)
    mr=np.cos(thetam)*np.cos(theta)+np.sin(thetam)*np.sin(theta)*np.cos(phi-Omega*tns)
    psi=phi-Omega*tns
    Br=2.*MB*(np.cos(thetam)*np.cos(theta)+np.sin(thetam)*np.sin(theta)*np.cos(psi))
    Btheta=MB*(np.cos(thetam)*np.sin(theta)-np.sin(thetam)*np.cos(theta)*np.cos(psi))
    Bphi=MB*np.sin(thetam)*np.sin(psi)
    Bx=Br*np.sin(theta)*np.cos(phi)+Btheta*np.cos(theta)*np.cos(phi)-Bphi*np.sin(phi)
    By=Br*np.sin(theta)*np.sin(phi)+Btheta*np.cos(theta)*np.sin(phi)+Bphi*np.cos(phi)
    Bz=Br*np.cos(theta)-Btheta*np.sin(theta)
    return Bx,By,Bz
def MagneticFieldSquared(r,theta,phi,B0,P,thetam,tns,rns): #[Gauss]
    Omega=(2.*np.pi)/P
    MB=0.5*B0*((rns/r)**3.)
    mr=np.cos(thetam)*np.cos(theta)+np.sin(thetam)*np.sin(theta)*np.cos(phi-Omega*tns)
    psi=phi-Omega*tns
    B2=(MB**2.)*(3.*(mr**2.)+1)
    return B2

########################################################################
### ************** PLASMA MASS IN THE NS MAGNETOSPHERE *****************
########################################################################
### WE USE THE ELECTRON/POSITRON DENSITY IN THE GJ MODEL
########################################################################
def PlasmaMass(r,theta,phi,B0,P,thetam,tns,rns): #[electronVolt]
    Omega=(2.*np.pi)/P
    mr=np.cos(thetam)*np.cos(theta)+np.sin(thetam)*np.sin(theta)*np.cos(phi-Omega*tns)
    MB=(0.5*B0)*((rns/r)**3.)
    Bz=MB*(3.*np.cos(theta)*mr-np.cos(thetam)) 
    plasmaMass=np.sqrt(2.*Omega*abs(Bz)*ebis/me)
    return plasmaMass

#############################################################################
### ************** AXION PHOTON CONVERSION PROBABILITY **********************
#############################################################################
#def probGRAPPA(g,B2,w,dwp,ma,Lc2):
########################
#### NEW DEFINITION ####
########################
def probGRAPPA(g,B2,w,dwp,ma,vc):
    p=(np.pi/2.)*((g**2)*B2)*(1/vc/abs(dwp)) # NEW EXPRESSION
    #p=(np.pi/2.)*((g**2.)*B2)*(w/(ma*dwp))
    #p=(np.pi/4.)*((g**2.)*B2)*Lc2*(w/np.sqrt(w**2.-ma**2.))  # CHRISTOPH WENIGER
    #p=(np.pi/4.)*((g**2.)*B2)*Lc2*(w*w/(w*w-ma*ma))         # MIKAEL LEROY
    return p

#############################################################
### ANALYTIC EXPRESSION FOR THE CONVERSION RADIUS WHERE WP=MA
#############################################################
def Rc(theta,phi,B0,P,thetam,ma,tns,rns): #[meters]
    Omega=(2.*np.pi)/P
    mr=np.cos(thetam)*np.cos(theta)+np.sin(thetam)*np.sin(theta)*np.cos(phi-Omega*tns)
    X3=(ebis*Omega*B0/(ma*ma*me))*abs((3.*np.cos(theta)*mr-np.cos(thetam)))
    # to change the resonant conversion condition to e.g. wp = 2*pi*ma 
    # one can simply replace ma by 2*pi*ma in the expression above of X3
    rc=rns*(abs(X3)**(1./3))
    return rc/dnu
############################################################
### FIND MAXIMUM CONVERSION RADIUS LOOKING AT ALL DIRECTIONS
############################################################
def Rcmax(B0,P,ma,thetam): #[meters]
    rns=1e4*dnu
    Omega=(2.*np.pi)/P
    theta=np.linspace(0.,np.pi,int(1e3))
    psi=np.linspace(0.,2.*np.pi,int(1e3))
    ttheta,ppsi = np.meshgrid(theta, psi, sparse=True)
    mr=np.cos(thetam)*np.cos(ttheta)+np.sin(thetam)*np.sin(theta)*np.cos(ppsi)
    e3=abs(3.*np.cos(theta)*mr-np.cos(thetam))
    X3=(ebis*Omega*B0/(ma*ma*me))*abs(e3).max()
    rc=rns*(abs(X3)**(1./3))
    return rc/dnu
####################################################################################
### MINIMAL DISTANCE BETWEEN TWO NEIGHBOURING SURFACES OF RESONANT CONVERSION AT THE 
### NEUTRON STAR'S SURFACE (OR alp*RNS) = ESTIMATE OF THE REQUIRED RESOLUTION FOR 
### INDIVIDUAL TRAJECTORIES
####################################################################################
def ThroatSize(B0,P,ma,x=1): #[meters]
    rns=1e4*dnu
    Omega=(2.*np.pi)/P
    thetam=0
    tns=0.
    d=-1
    factor=(ma*ma*me*(x**3))/(ebis*Omega*B0)
    if factor<=1:
        costheta=np.sqrt((1./3.)*(1+factor))
        theta1,theta2=np.arccos(costheta),np.arccos(-costheta)
        costheta=np.sqrt((1./3.)*(1-factor))
        theta3,theta4=np.arccos(costheta),np.arccos(-costheta)
        phi=0
        r1=Rc(theta1,phi,B0,P,thetam,ma,tns,rns)
        r2=Rc(theta2,phi,B0,P,thetam,ma,tns,rns)
        r3=Rc(theta3,phi,B0,P,thetam,ma,tns,rns)
        r4=Rc(theta4,phi,B0,P,thetam,ma,tns,rns)
        u1,v1=r1*np.sin(theta1),r1*np.cos(theta1)
        u2,v2=r2*np.sin(theta2),r2*np.cos(theta2)
        u3,v3=r3*np.sin(theta3),r3*np.cos(theta3)
        u4,v4=r4*np.sin(theta4),r4*np.cos(theta4)
        d=np.sqrt((u1-u3)**2+(v1-v3)**2)
    return d

#############################################################################
##################### TRAJECTORIES OF PHOTONS ###############################
#############################################################################

##################################################
### TRAJECTORY OF TYPE traj
### "classical" (straight lines)
### "geodesics" (geodesics in a schwarschild metric)
##################################################
def Trajectory(x0,y0,z0,vx0,vy0,vz0,ri,thetai,phii,nsteps,traj):
    eR=np.array([np.sin(thetai)*np.cos(phii),np.sin(thetai)*np.sin(phii),np.cos(thetai)])
    eTheta=np.array([np.cos(thetai)*np.cos(phii),np.cos(thetai)*np.sin(phii),-np.sin(thetai)])
    ePhi=np.array([-np.sin(phii),np.cos(phii),0])
    xi,yi,zi=spher_to_cart(ri,thetai,phii)
    ux,uy,uz=eR
    OP=np.array([xi,yi,zi])
    OM=np.array([x0,y0,z0])
    PM=OM-OP
    A=PM[0]*ePhi[0]+PM[1]*ePhi[1]+PM[2]*ePhi[2]
    B=-(PM[0]*eTheta[0]+PM[1]*eTheta[1]+PM[2]*eTheta[2])
    b=np.sqrt(A**2+B**2)
    alpha=np.arctan2(B,A)
    ROT=ROTMAT(ux,uy,uz,alpha-np.pi/2.)
    if traj=="classical":
        u,v = trajectoryCL(b,nsteps,ri)
    if traj=="geodesics":
        u,v = trajectoryGR(b,nsteps,ri,params["M"])
    Xx = u*np.sin(thetai)*np.cos(phii)+v*(-np.cos(thetai)*np.cos(phii))
    Yy = u*np.sin(thetai)*np.sin(phii)+v*(-np.cos(thetai)*np.sin(phii))
    Zz = u*np.cos(thetai)+v*np.sin(thetai)
    x=ROT[0][0]*Xx+ROT[0][1]*Yy+ROT[0][2]*Zz
    y=ROT[1][0]*Xx+ROT[1][1]*Yy+ROT[1][2]*Zz
    z=ROT[2][0]*Xx+ROT[2][1]*Yy+ROT[2][2]*Zz
    return x,y,z
#######################################
##### NULL SCHWARSCHILD GEODESICS
##### IN TWO DIMENSIONS (VARIABLES U,V)
#######################################
def funcSCH(t,y,M):
    r,dr,phi,dphi=y 
    d2r = ((3*M)/(r*(r-2*M)))*(dr**2) - (M/r**2)*(1-(2*M)/r) + (r-2*M)*(dphi**2)
    d2phi = -(2/r)*dr*dphi*(r-3*M)/(r-2*M)
    return np.array([dr,d2r,dphi,d2phi])
def trajectoryGR(b,nsteps,ri,M):
    u0,v0=ri,b
    dt = (2.*ri)/(c*nsteps)
    vx0,vy0,vz0=-c,0.,0.
    x0,y0,z0=u0,v0,0.
    r0,theta0,phi0=cart_to_spher(x0,y0,z0)
    M=M/mnu  # NS mass in kg
    M=M*G/c2 # NS mass in geometrical units
    vt0=1
    gtt=-(1-(2*M)/r0)
    grr=1/(1-(2*M)/r0)
    gphiphi=r0**2
    Vt0=vt0/np.sqrt(abs(gtt))
    Vr0=(vx0*np.cos(phi0)+vy0*np.sin(phi0))/np.sqrt(abs(grr))
    Vphi0=(-vx0*np.sin(phi0)+vy0*np.cos(phi0))/np.sqrt(abs(gphiphi))
    dr0=Vr0/Vt0
    dphi0=Vphi0/Vt0
    Y0=[r0,dr0,phi0,dphi0]
    y=[Y0]
    system=scipy.integrate.ode(funcSCH).set_integrator(name="dopri5")
    system.set_initial_value(Y0,0).set_f_params(M)
    k=0
    while system.successful() and k < nsteps:
        system.integrate(system.t+dt)
        y.append(system.y)
        k+=1
    y=np.array(y)
    R,DR,PHI,DPHI=y[:,0],y[:,1],y[:,2],y[:,3]
    u=R*np.cos(PHI)
    v=R*np.sin(PHI)
    return u,v
############################################
####### STRAIGHT LINE TRAJECTORIES #########
############################################
def trajectoryCL(b,nsteps,ri):
    v0,u0 = b, ri
    dt = (2.*ri)/(c*nsteps)
    u = u0 - c*np.linspace(0.,nsteps*dt,nsteps,endpoint=True)
    v = v0*np.ones(nsteps)
    return u,v
def trajectory(b,nsteps,ri,M,traj):
    if traj=="classical":
        u,v = trajectoryCL(b,nsteps,ri)
    if traj=="geodesics":
        u,v = trajectoryGR(b,nsteps,ri,M)
    return u,v

############################################################################################
############################################################################################
def pixelValue(X,Y,Z,nsteps,ri,resH,B0,P,thetam,ma,g,tns,rns,M,rhoDMinf,vDMdisp,vDMinf):
    ################################################
    ### THIS CUTS TRAJECTORIES THAT CROSS THE NS
    ### IF i IS THE MIN. INTEGER SUCH THAT R[i]<RNS 
    ### THEN ONLY POINTS UNTIL i (INCLUDED) ARE KEPT
    ################################################
    mask=np.sqrt(X**2+Y**2+Z**2)<(rns/dnu)
    i=np.argmax(mask)
    if i>0:
        X,Y,Z=X[:i+1],Y[:i+1],Z[:i+1]
    #######################################################
    #######################################################
    #### ROOT FINDING (FIND RESONANT CONVERSION POINTS) ###
    ##############################################################################
    ### METHOD 1. INCREASES RESOLUTION LOCALLY CLOSE TO RESONANT CONVERSION POINTS
    ##############################################################################
    res=ThroatSize(B0,P,ma)
    resI=2.*ri/nsteps
    R,THETA,PHI=cart_to_spher(X*dnu,Y*dnu,Z*dnu)
    Wp=PlasmaMass(R,THETA,PHI,B0,P,thetam,tns,rns)
    D=Wp-ma
    positive=D>0
    igmax=0
    try:
        igmax=np.argmax(positive)
    except:
        pass
    if igmax>0:
        I=[igmax]
    else:
        I=[]
    T=np.linspace(0,len(X)-1,num=len(X),endpoint=True)
    x=interpolate.interp1d(T,X)
    y=interpolate.interp1d(T,Y)
    z=interpolate.interp1d(T,Z)
    ROOTS=[]
    try:
        i=I[0]
        d=2
        num=max(1,int(2.*d*resI/resH)) # NEW RESOLUTION IS resH
        jmin=max(0,i-d)
        jmax=min(jmin+2*d,len(X)-1)
        TH=np.linspace(jmin,jmax,num=num)
        XH,YH,ZH=x(TH),y(TH),z(TH)
        RH,THETAH,PHIH=cart_to_spher(XH*dnu,YH*dnu,ZH*dnu)
        WpH=PlasmaMass(RH,THETAH,PHIH,B0,P,thetam,tns,rns)
        DH=WpH-ma
        positive=DH>0
        I2=1+np.where(np.bitwise_xor(positive[1:], positive[:-1]))[0]
        if len(I2)>0:
            i2=I2[0]
            roots=[RH[i2-1],THETAH[i2-1],PHIH[i2-1],RH[i2],THETAH[i2],PHIH[i2]]
            ROOTS+=[roots]
    except:
        pass
    ##############################################################################
    ### METHOD 2. TRIES TO FIND THE ROOTS WITHOUT INCREASING THE RESOLUTION
    ##############################################################################
    #S=np.cumsum(np.sqrt((X[1:]-X[:-1])**2+(Y[1:]-Y[:-1])**2+(Z[1:]-Z[:-1])**2))
    #S=np.append(S[::-1],0)[::-1]
    #R,THETA,PHI=cart_to_spher(X*dnu,Y*dnu,Z*dnu)
    #Wp=PlasmaMass(R,THETA,PHI,B0,P,thetam,tns,rns)
    #D=Wp-ma
    #positive=D>0
    #I2=1+np.where(np.bitwise_xor(positive[1:], positive[:-1]))[0]
    #ROOTS=[]
    #for i2 in I2:
    #    roots=[R[i2-1],THETA[i2-1],PHI[i2-1],
    #           R[i2],THETA[i2],PHI[i2]]
    #    ROOTS+=[roots]
    ##################################################
    ##################################################
    RC1,RC2=[],[]
    THETAC1,THETAC2=[],[]
    PHIC1,PHIC2=[],[]
    try:
        roots=ROOTS[0]
        rt2,thetat2,phit2=roots[3],roots[4],roots[5]
        rt1,thetat1,phit1=roots[0],roots[1],roots[2]
        RC1=[rt1]
        RC2=[rt2]
        THETAC1=[thetat1]
        THETAC2=[thetat2]
        PHIC1=[phit1]
        PHIC2=[phit2]
    except:
        pass
    pixelInfo={"RC1":RC1,"RC2":RC2,"THETAC1":THETAC1,"THETAC2":THETAC2,"PHIC1":PHIC1,"PHIC2":PHIC2}
    return pixelInfo

#############################################################################
###### RADIATED POWER PER UNIT SOLID ANGLE USING RAY-TRAYCING METHOD ########
#############################################################################
def SIGNAL(Lgrid,resGrid,resI,resH,
           B0,P,ma,thetam,
           thetai,phii,tns):
    ngrid=int(Lgrid/resGrid)
    nsteps=int(2.*Rcmax(B0,P,ma,thetam)/resI)
    ################################
    #### OPTIONAL DEFAULT PARAMETERS
    ################################
    traj="classical"
    g=1e-21
    rns=1e4*dnu
    M=1.*M0*mnu
    rhoDMinf=0.3*1e9*1e6/(dnu**3)
    vDMdisp=200*1e3*vnu
    vDMinf=250*1e3*vnu
    ################################
    ################################
    Grid=np.zeros(shape=(ngrid,ngrid))
    dPd0=0. 
    rcmax=1.*Rcmax(B0,P,ma,thetam)
    if rcmax>=(rns/dnu):
        ti=time.perf_counter()
        ##################################################
        ##################################################
        ri=Rcmax(B0,P,ma,thetam)
        print("********************************************")
        print("GRID RESOLUTION\n",round(Lgrid/ngrid,2),"m")
        print("INITIAL TRAJECTORY RESOLUTION\n",round(2.*ri/nsteps,2),"m")
        ##################################################
        ##################################################
        eR=np.array([np.sin(thetai)*np.cos(phii),np.sin(thetai)*np.sin(phii),np.cos(thetai)])
        eTheta=np.array([np.cos(thetai)*np.cos(phii),np.cos(thetai)*np.sin(phii),-np.sin(thetai)])
        ePhi=np.array([-np.sin(phii),np.cos(phii),0])
        I,J=[],[]
        RC1,RC2=[],[]
        THETAC1,THETAC2=[],[]
        PHIC1,PHIC2=[],[]
        GridInfo={}
        for i in range(ngrid):
            for j in range(ngrid):

                A=Lgrid*(-0.5+i/(ngrid-1.))
                B=Lgrid*(-0.5+j/(ngrid-1.))
                x0,y0,z0=ri*eR + A*ePhi + B*(-eTheta)
                vx0,vy0,vz0=-c*eR
                X,Y,Z=Trajectory(x0,y0,z0,vx0,vy0,vz0,ri,thetai,phii,nsteps,traj)

                pixelInfo=pixelValue(X,Y,Z,nsteps,ri,resH,B0,P,thetam,ma,g,tns,rns,M,rhoDMinf,vDMdisp,vDMinf)

                RC1+=pixelInfo["RC1"]
                RC2+=pixelInfo["RC2"]
                THETAC1+=pixelInfo["THETAC1"]
                THETAC2+=pixelInfo["THETAC2"]
                PHIC1+=pixelInfo["PHIC1"]
                PHIC2+=pixelInfo["PHIC2"]
                nroots=len(pixelInfo["RC1"])

                I+=[i]*nroots
                J+=[j]*nroots

        I=np.array(I)
        J=np.array(J)

        RC1=np.array(RC1)
        RC2=np.array(RC2)
        THETAC1=np.array(THETAC1)
        THETAC2=np.array(THETAC2)
        PHIC1=np.array(PHIC1)
        PHIC2=np.array(PHIC2)
    
        X1,Y1,Z1=spher_to_cart(RC1,THETAC1,PHIC1)
        X2,Y2,Z2=spher_to_cart(RC2,THETAC2,PHIC2)
        XT,YT,ZT=0.5*(X1+X2),0.5*(Y1+Y2),0.5*(Z1+Z2)
        RT,THETAT,PHIT=cart_to_spher(XT,YT,ZT)
    
        VXT,VYT,VZT=X2-X1,Y2-Y1,Z2-Z1
        V=np.sqrt(VXT**2+VYT**2+VZT**2)
        VXT=VXT/V
        VYT=VYT/V
        VZT=VZT/V

        BX,BY,BZ=MagneticField(RT,THETAT,PHIT,B0,P,thetam,tns,rns)
        B2=BX**2+BY**2+BZ**2
        Bperp2=B2-(BX*VXT+BY*VYT+BZ*VZT)**2

        B2=Bperp2
        
        DS=np.sqrt((X2-X1)**2+(Y2-Y1)**2+(Z2-Z1)**2)
        WP1=PlasmaMass(RC1,THETAC1,PHIC1,B0,P,thetam,tns,rns)
        WP2=PlasmaMass(RC2,THETAC2,PHIC2,B0,P,thetam,tns,rns)
        DWPDS=(1./DS)*(WP2-WP1)  # estimation of plasma mass derivative for conversion probability
        VDM=np.sqrt(2.*Gbis*M/RT)   
        RHODM=Density(RT,M,rhoDMinf,vDMdisp)    # dark matter density based on Liouville's theorem
        #gammai=1./np.sqrt(1-VDM**2)
        # SET GAMMA = 1
        gammai=1.
        w=ma*gammai
        PROB=probGRAPPA(g,B2,w,DWPDS,ma,VDM)
        PIXEL=PROB*VDM*RHODM/(4.*np.pi)
        PIXEL=2*PIXEL # ad-hoc multiplication by 2 two account for "reflexion"

        mask=(RT>=rns)

        PIXEL=PIXEL[mask]
        I=I[mask]
        J=J[mask]
        RT=RT[mask]
        THETAT=THETAT[mask]
        PHIT=PHIT[mask]
        
        for p in range(len(I)):
            i,j=I[p],J[p]
            Grid[j][i]+=PIXEL[p]

        dPd0=Grid.sum()
        d=Lgrid/ngrid
        dPd0=dPd0*(d*dnu)*(d*dnu)
        #GridInfo={"RC1":RC1,"RC2":RC2,"THETAC1":THETAC1,"THETAC2":THETAC2,"PHIC1":PHIC1,"PHIC2":PHIC2,"I":I,"J":J}
        GridInfo={"RT":RT,"THETAT":THETAT,"PHIT":PHIT,"PIXEL":PIXEL}
        tf=time.perf_counter()
        print("COMPUTING TIME\n",round(tf-ti,2),"s")
        print("********************************************")
    return dPd0,Grid,GridInfo

##################################################################################
### THEORETICAL PREDICTION FOR THE RADIATED POWER PER UNIT SOLID ANGLE
### A. Hook, Y. Kahn, B. R. Safdi, and Z. Sun, 
### Radio Signals from Axion Dark Matter Conversion in Neutron Star Magnetospheres
### Phys. Rev. Lett. 121, 241102 (2018), arXiv:1804.03145 [hep-ph].
##################################################################################

####################################################
### CONVERSION PROBABILITY OF AN AXION INTO A PHOTON
####################################################
def probBEN(theta,phi,B0,P,thetam,ma,g,tns,rns): #[dimensionless]
    Omega=(2.*np.pi)/P
    rc=Rc(theta,phi,B0,P,thetam,ma,tns,rns)*dnu
    mr=np.cos(thetam)*np.cos(theta)+np.sin(thetam)*np.sin(theta)*np.cos(phi-Omega*tns)
    B2=(1.+3.*mr*mr)*((0.5*B0)**2.)*(rns/rc)**6.
    prob=(np.pi/3.)*rc*B2*(g*g)/ma
    return prob
####################################################################
### ANALYTICAL ESTIMATE FOR THE RADIATED POWER PER UNIT SOLIDE ANGLE
####################################################################
def signalBEN(theta,phi,B0,P,thetam,ma,g,tns,
                rns=1e4*dnu,M=1.*M0*mnu,
                rhoDMinf=0.3*1e9*1e6/(dnu**3),vDMdisp=200*1e3*vnu): #[eV to some power]
    rc=Rc(theta,phi,B0,P,thetam,ma,tns,rns)*dnu
    vc=np.sqrt(2*Gbis*M/rc)
    prob=probBEN(theta,phi,B0,P,thetam,ma,g,tns,rns)
    signal=2.*prob*Density(rc,M,rhoDMinf,vDMdisp)*vc*(rc**2)
    return signal


######################################################
### SWITCH BETWEEN CARTESIAN AND SPHERICAL COORDINATES
######################################################
def cart_to_spher(x,y,z):
    r=np.sqrt(x**2+y**2+z**2)
    theta=np.arccos(z/r)
    phi=np.arctan2(y,x)
    try: 
        if phi<0.:
            phi+=2.*np.pi
    except:
        mask=phi<0.
        phi[mask]+=2.*np.pi
    return r,theta,phi
def spher_to_cart(r,theta,phi):
    x=r*np.sin(theta)*np.cos(phi)
    y=r*np.sin(theta)*np.sin(phi)
    z=r*np.cos(theta)
    return x,y,z

