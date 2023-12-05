import numpy as np   # scientific computing tools 
import nutopy as nt  # indirect methods and homotopy
import plottings     # for plots

# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# Function to computes self-intersections of a curve in a 2-dimensional space
def get_self_intersections(curve):
    n = curve[:,0].size
    intersections = []
    for i in range(n-3):
        A = curve[i,:]
        B = curve[i+1,:]
        for j in range(i+2,n-1):
            C = curve[j,:]
            D = curve[j+1,:]
            # Matrice M : M z = b
            m11 = B[0] - A[0]
            m12 = C[0] - D[0]
            m21 = B[1] - A[1]
            m22 = C[1] - D[1]
            det = m11*m22-m12*m21
            if(np.abs(det)>1e-8):
                b1 = C[0] - A[0]
                b2 = C[1] - A[1]
                la = (m22*b1-m12*b2)/det
                mu = (m11*b2-m21*b1)/det
                if(la>=0. and la<=1. and mu>=0. and mu<=1.):
                    xx = {'i': i, 'j': j, \
                          'x': np.array(A + la * (B-A)), \
                          'la': la, 'mu': mu}
                    intersections.append(xx)
    return intersections

# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# get the first self-intersection of the curve together with the values of
# the parameter α
def get_αs_splitting(curve, αs):
    xxs = get_self_intersections(curve)
    xx  = xxs[0]
    x   = xx.get('x')
    i   = xx.get('i')
    j   = xx.get('j')
    la  = xx.get('la')
    mu  = xx.get('mu')
    α1  = αs[i]+la*(αs[i+1]-αs[i])
    α2  = αs[j]+mu*(αs[j+1]-αs[j])
    return α1, α2, x

# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# construct the extremal mapping
def make__extremal(H):
    return nt.ocp.Flow(H)
    
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# construct the covector mapping which maps α to the initial covector of norm 1
def make__covector(g): 
    
    def covector(q, α):
        g1, g2 = g(q)
        p0 = np.array([np.sin(α)*np.sqrt(g1), 
                       np.cos(α)*np.sqrt(g2)])
        return p0
        
    def dcovector(q, α, dα):
        g1, g2 = g(q)
        dp0 = np.array([ np.cos(α)*np.sqrt(g1)*dα, 
                        -np.sin(α)*np.sqrt(g2)*dα])
        return dp0
        
    def d2covector(q, α, dα, d2α):
        g1, g2 = g(q)
        d2p0 = np.array([-np.sin(α)*np.sqrt(g1)*dα*d2α, 
                         -np.cos(α)*np.sqrt(g2)*dα*d2α])
        return d2p0
        
    covector = nt.tools.tensorize(dcovector, d2covector, \
                                       tvars=(2,))(covector)
    
    return covector

# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# construct the geodesic mapping parameterized by α0 to time t
# the initial point and time are given
def make__geodesic(t0, q0, covector, extremal, N=100):

    def tspan(t0, t):
        return list(np.linspace(t0, t, N))
    
    def geodesic(t, α0):
        p0     = covector(q0, α0)
        q, p   = extremal(t0, q0, p0, tspan(t0, t))
        return q

    return geodesic
        
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# construct the mapping which maps α to the initial state-costate point
def make__z0(q0, covector):
    
    def z0(α):
        p0 = covector(q0, α)
        return np.concatenate([q0, p0])

    return z0

# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# construct the function to compute conjugate points
def make__conjugate(t0, q0, covector, extremal, H):

    # Jacobi field: dz(t, p(α0), dp(α0))
    @nt.tools.vectorize(vvars=(1,))
    def jacobi(t, α0):
        p0, dp0  = covector(q0, (α0, 1.))
        (q, dq), (p, dp) = extremal(t0, q0, (p0, dp0), t)
        return (q, dq), (p, dp)

    # Derivative of dq w.r.t. t and α0
    def djacobi(t, α0):
        #
        p0, dp0, d2p0 = covector(q0, (α0, 1., 1.))
        #
        (q, dq1, d2q), (p, dp1, _) = extremal(t0, q0, \
                                            (p0, dp0, dp0), t)
        (q, dq2), (p, dp2)         = extremal(t0, q0, \
                                              (p0, d2p0), t)
        #
        hv, dhv   = H.vec(t, (q, dq1), (p, dp1))
        #
        ddqda     = d2q+dq2  # ddq/dα
        ddqdt     = dhv[0:2] # ddq/dt
        return (q, dq1), (p, dp1), (ddqdt, ddqda)

    # Function to compute conjugate time together with 
    # the initial angle and the associated conjugate point
    #
    # conjugate(tc, qc, a0) = ( det( dq(tc, a0), 
    #                                Hv(tc, z(tc, a0)) ), 
    #                        qc - pi_q(z(tc, a0)) ),
    #
    # where pi_q(q, p) = q and 
    # z(t, a) = extremal(t0, q0, p(a), t).
    #
    # Remark: y = (tc, qc)
    #
    def conjugate(y, a):
        tc     = y[0]
        qc     = y[1:3]
        α0 = a[0]
        #
        (q, dq), (p, dp) = jacobi(tc, α0)
        hv     = H.vec(tc, q, p)[0:2]
        #
        c      = np.zeros(3)
        c[0]   = np.linalg.det([hv, dq]) / tc
        c[1:3] = qc - q
        return c

    # Derivative of conjugate
    def dconjugate(y, a):
        tc = y[0]
        qc = y[1:3]
        α0 = a[0]
        #
        (q, dq), (p, dp), (ddqdt, ddqda) = djacobi(tc, α0)
        #
        # dc/da
        hv, dhv     = H.vec(tc, (q, dq), (p, dp))
        dcda        = np.zeros((3, 1))
        dnum        = np.linalg.det([dhv[0:2], dq]) + \
        np.linalg.det([hv[0:2], ddqda]) 
        dcda[0,0]   = dnum/tc
        dcda[1:3,0] = -dq
        #
        # dc/dy = (dc/dt, dc/dq)
        hv, dhv     = H.vec(tc, (q, hv[0:2]), (p, hv[2:4]))
        dcdy        = np.zeros((3, 3))
        num         = np.linalg.det([hv[0:2], dq])
        dnum        = np.linalg.det([dhv[0:2], dq]) + \
        np.linalg.det([hv[0:2], ddqdt]) 
        dcdy[0,0]   = (dnum*tc-num)/tc**2
        dcdy[1:3,0] = -hv[0:2]
        dcdy[1,1]   = 1.
        dcdy[2,2]   = 1.
        return dcdy, dcda

    return conjugate, dconjugate

# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# construct a function to compute a part of the conjugate locus
def make__get_part_of_conjugate_locus(t0, q0, covector, extremal, conjugate, dconjugate):

    def get_part_of_conjugate_locus(α0, αf):
    
        # -------------------------
        # Get first conjugate point
    
        # Initial guess
        tci    = np.pi
        α      = [α0]
        p0     = covector(q0, α[0])
        xi, pi = extremal(t0, q0, p0, tci)
    
        yi      = np.zeros(3)
        yi[0]   = tci
        yi[1:3] = xi
    
        # Equations and derivative
        fun   = lambda t: conjugate(t, α)
        dfun  = lambda t: dconjugate(t, α)[0]
    
        # Callback
        def print_conjugate_time(infos):
            print('    Conjugate time estimation: \
            tc = %e for α = %e' % (infos.x[0], α[0]), end='\r')
    
        # Options
        opt  = nt.nle.Options(Display='on')
    
        # Conjugate point calculation for 
        # initial homotopic parameter
        print(' > Get first conjugate time and point:\n')
        sol   = nt.nle.solve(fun, yi, df=dfun, \
                             callback=print_conjugate_time, \
                             options=opt)
    
        # -------------------
        # Get conjugate locus
    
        # Options
        opt = nt.path.Options(MaxStepSizeHomPar=0.05, \
                              Display='on');
    
        # Initial solution
        y0 = sol.x
    
        # Callback
        def progress(infos):
            current   = infos.pars[0]-α0
            total     = αf-α0
            barLength = 50
            percent   = float(current * 100.0 / total)
            arrow = '-' * int(percent/100 * barLength - 1) + \
            '>'
            spaces    = ' ' * (barLength - len(arrow))
    
            print('    Progress: [%s%s] %1.2f %%' % \
                  (arrow, spaces, round(percent, 2)), end='\r')
    
        # Conjugate locus calculation
        print('\n\n > Get the conjugate locus for α in \
        [%e, %e]:\n' % (α0, αf))
        sol = nt.path.solve(conjugate, y0, α0, αf, \
                            options=opt, df=dconjugate)
        print('\n')
    
        return sol.xout

    return get_part_of_conjugate_locus

# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# construct a function to plot a given geodesic
def make_plot_geodesic():

    def plot_geodesic(α, tf=None, stop='equator_2'):
        
        if 

    return plot_geodesic
    
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
class Problem:

    def __init__(self, H, g, t0, q0, data, N=100):

        self.t0 = t0     # initial time
        self.q0 = q0     # initial point
        self.data = data # data to save computations
        self.H = H       # Hamiltonian
        self.g = g       # metric
        self.N = N       # number of time steps when computing geodesics
        
        # Hamiltonian exponential map and its derivatives
        self.extremal = make__extremal(H)
        
        # Initial covector parameterization and its derivatives up to order 2
        self.covector = make__covector(g)
        
        # geodesic
        self.geodesic = make__geodesic(self.t0, self.q0, self.covector, self.extremal, self.N)

        # initial z0(α)
        self.z0 = make__z0(self.q0, self.covector)

        # Function to plot geodesics
        self.plot_geodesics = plottings.Geodesics(self.geodesic, self.H, self.z0, self.extremal)
        
        # conjugate function used to compute conjugate points
        self.conjugate, self.dconjugate = make__conjugate(self.t0, self.q0, \
                                                          self.covector, self.extremal, self.H)
        
        # function to compute part of the conjugate locus
        self.get_part_of_conjugate_locus = make__get_part_of_conjugate_locus(self.t0, self.q0, \
                                                                             self.covector, \
                                                                             self.extremal, \
                                                                             self.conjugate, \
                                                                             self.dconjugate)
        
        # Function to compute the conjugate locus
        self.conjugate_locus = None # first initialization
        self.plot_conjugate_locus = None # first initialization
        self.compute_conjugate_locus = self.__compute_conjugate_locus(self.get_part_of_conjugate_locus, \
                                                                      self.data)

    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
    # general plot function
    def plot(self, fig=None, coords=plottings.Coords.SPHERE, geodesics=0, \
             conjugate=False, cut=False, wavefronts=False, tf=None, azimuth=140):

        if geodesics>0:
            if not(cut):
                cut_time_fun = None
            else:
                cut_time_fun = self.cut_time
            fig = self.plot_geodesics(coords=coords, fig=fig, nb_geodesics=geodesics, \
                               ratio_tf=1, cut=cut_time_fun, recompute=True, \
                               azimuth=azimuth, tf=tf)
            
        if conjugate:
            if not self.plot_conjugate_locus is None:
                self.plot_conjugate_locus(coords=coords, fig=fig, azimuth=azimuth)
            else:
                print('You must compute the conjugate locus before plotting it.')

        if cut:
            pass
            #self.splitting_plot(coords=coords, fig=fig, azimuth=azimuth)

        if wavefronts:
            pass
            #self.wavefronts_plot(coords=coords, fig=fig, azimuth=azimuth)

        return fig

    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
    # this constructor is internal since the constructed function modifies
    # self.conjugate_locus and self.plot_conjugate_locus
    def __compute_conjugate_locus(self, get_part_of_conjugate_locus, data):
    
        def __get_conjugate_locus(restart):
    
            def get_conjugate_locus_right(gap):
                α0  = -np.pi/2+gap
                αf  =  np.pi/2-gap
                sol = get_part_of_conjugate_locus(α0, αf)
                return sol
            
            def get_conjugate_locus_left(gap):
                α0  = 1*np.pi/2+gap
                αf  = 3*np.pi/2-gap
                sol = get_part_of_conjugate_locus(α0, αf)
                return sol
        
            # Get conjugate locus
            if data.contains('conjugate_locus') and not restart:
                conjugate_locus_list = data.get('conjugate_locus');
                conjugate_locus = {
                    "left":  np.array(conjugate_locus_list["left"]),
                    "right": np.array(conjugate_locus_list["right"]),
                }
                print('Conjugate locus loaded')
            else:
                gap = 1e-2
                conjugate_locus_left  = get_conjugate_locus_left(gap)
                conjugate_locus_right = get_conjugate_locus_right(gap)
                conjugate_locus = {
                    "left": conjugate_locus_left,
                    "right": conjugate_locus_right
                }
                conjugate_locus_list = {
                    "left":  conjugate_locus["left"].tolist(),
                    "right": conjugate_locus["right"].tolist(),
                }
                data.update({'conjugate_locus':conjugate_locus_list})
                print('Conjugate locus saved')
    
            return conjugate_locus
        
        def compute_conjugate_locus(restart=False):
            # compute
            conjugate_locus = __get_conjugate_locus(restart)
            # save in prob
            self.conjugate_locus = conjugate_locus
            self.plot_conjugate_locus = plottings.Conjugate_Locus(conjugate_locus)
            # return the conjugate locus
            return conjugate_locus
    
        return compute_conjugate_locus




    

