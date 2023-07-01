import numpy as np   # scientific computing tools 
import nutopy as nt  # indirect methods and homotopy
import plottings     # for plots

# Function to computes self-intersections 
# of a curve in a 2-dimensional space
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

#
def get_αs_splitting(curve, αs):
    #
    xxs       = get_self_intersections(curve)
    xx        = xxs[0]
    x         = xx.get('x')
    i         = xx.get('i')
    j         = xx.get('j')
    la        = xx.get('la')
    mu        = xx.get('mu')
    α1    = αs[i]+la*(αs[i+1]-αs[i])
    α2    = αs[j]+mu*(αs[j+1]-αs[j])
    return α1, α2, x

class Problem:

    def __init__(self, H, g, t0, q0, data, restart=False):

        self.t0 = t0
        self.q0 = q0
        self.data = data
        
        # Hamiltonian
        self.H = H
        
        # metric
        self.g = g
        
        # Hamiltonian exponential map and its derivatives
        extremal = nt.ocp.Flow(H)
        self.extremal = extremal
        
        # Initial covector parameterization 
        # and its derivatives up to order 2
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
        self.covector = covector
        
        # Compute a geodesic parameterized by α0 to time t
        # The initial point is fixed to q0 and t0 = 0
        def tspan(t0, t):
            N      = 100
            return list(np.linspace(t0, t, N))
        
        def geodesic(t, α0):
            p0     = covector(q0, α0)
            q, p   = extremal(t0, q0, p0, tspan(t0, t))
            return q
        
        def z0(α0):
            p0 = covector(q0, α0)
            return np.concatenate([q0, p0])
            
        # Function to plot geodesics
        self.geodesics_plot = plottings.Geodesics(geodesic, H, \
                                                  z0, extremal)
        
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

        def get_conjugate_locus(α0, αf):
        
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

        def get_conjugate_locus_right(gap):
            α0  = -np.pi/2+gap
            αf  =  np.pi/2-gap
            sol = get_conjugate_locus(α0, αf)
            return sol
        
        def get_conjugate_locus_left(gap):
            α0  = 1*np.pi/2+gap
            αf  = 3*np.pi/2-gap
            sol = get_conjugate_locus(α0, αf)
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
        
        # Function to plot the conjugate locus
        conjugate_plot = plottings.Conjugate_Locus(conjugate_locus)
        self.conjugate_plot = conjugate_plot

        # Equation to calculate wavefronts
        def wavefront_eq(q, α0, tf):
            p0    = covector(q0, α0[0])
            qf, _ = extremal(t0, q0, p0, tf)
            return q - qf
        
        # Derivative
        def dwavefront_eq(q, dq, α0, dα0, tf):
            p0, dp0      = covector(q0, (α0[0], dα0[0]))
            (qf, dqf), _ = extremal(t0, q0, (p0, dp0), tf)
            return q-qf, dq - dqf
        
        wavefront_eq = nt.tools.tensorize(dwavefront_eq, \
                                          tvars=(1, 2), \
                                          full=True)(wavefront_eq)

        # Function to compute wavefront at time tf, q0 being fixed
        def get_wavefront(tf, α0, αf):
        
            # Options
            opt = nt.path.Options(Display='off', \
                                  MaxStepSizeHomPar=0.05, \
                                  MaxIterCorrection=10);
        
            # Initial solution
            p0     = covector(q0, α0)
            xf0, _ = extremal(t0, q0, p0, tf)
        
            # callback
            def progress(infos):
                current   = infos.pars[0]-α0
                total     = αf-α0
                barLength = 50
                percent   = float(current * 100.0 / total)
                arrow = '-' * int(percent/100.0 * barLength - 1)+ \
                '>'
                spaces    = ' ' * (barLength - len(arrow))
        
                print('    Progress: [%s%s] %1.2f %%' % \
                      (arrow, spaces, round(percent, 2)), end='\r')
        
            # wavefront computation
            print('\n > Get wavefront for tf =', tf, '\n')
            sol = nt.path.solve(wavefront_eq, xf0, α0, αf, \
                                args=tf, options=opt, \
                                df=wavefront_eq, callback=progress)
            print('\n')
        
            wavefront = (sol.xout, sol.parsout, tf)
        
            return wavefront

        def get_wavefront_right(tf, gap):
            α0  = -np.pi/2+gap
            αf  =  np.pi/2-gap
            sol = get_wavefront(tf, α0, αf)
            return sol
        
        def get_wavefront_left(tf, gap):
            α0  = 1*np.pi/2+gap
            αf  = 3*np.pi/2-gap
            sol = get_wavefront(tf, α0, αf)
            return sol
    
        # Get wavefronts
        if data.contains('wavefronts') and not restart:
            wavefronts = data.get('wavefronts')
            print('Wavefronts loaded')
        else:
            wavefronts = []
            gap = 1e-2
            #
            tf = 2.7
            wavefront_left  = get_wavefront_left(tf, gap)
            wavefront_right = get_wavefront_right(tf, gap)
            wavefront = {
                "left": (wavefront_left[0].tolist(), \
                         wavefront_left[1].tolist(), \
                         wavefront_left[2]),
                "right": (wavefront_right[0].tolist(), \
                          wavefront_right[1].tolist(), \
                          wavefront_right[2])
            }
            wavefronts.append(wavefront)
            #
            data.update({'wavefronts':wavefronts})
            print('wavefronts saved')
        
        # Function to plot wavefronts
        wavefronts_plot = plottings.Wavefronts(wavefronts)
        self.wavefronts_plot = wavefronts_plot

        # We compute one self-intersection of the wavefront
        wavefront = wavefronts[0]
        #
        wa_l = wavefront["left"]
        curve_l = np.array(wa_l[0])
        αs_l    = np.array(wa_l[1])
        tf_l    = wa_l[2]
        α1_l, α2_l, q_l = get_αs_splitting(curve_l, αs_l)
        #print('Self-intersection of the wavefront for tf =', tf_l)
        #print('α1 =', α1_l)
        #print('α2 =', α2_l)
        
        #
        wa_r = wavefront["right"]
        curve_r = np.array(wa_r[0])
        αs_r    = np.array(wa_r[1])
        tf_r    = wa_r[2]
        α1_r, α2_r, q_r = get_αs_splitting(curve_r, αs_r)
        #print('Self-intersection of the wavefront for tf =', tf_r)
        #print('α1 =', α1_r)
        #print('α2 =', α2_r)

        # Equations to compute Split(q0)
        def split_eq(y, α2):
            # y = (t, α1, q)
            t     = y[0]
            a1    = y[1]
            q     = y[2:4]
            a2    = α2[0]
            q1, _ = extremal(t0, q0, covector(q0, a1), t)
            q2, _ = extremal(t0, q0, covector(q0, a2), t)
            eq    = np.zeros(4)
            eq[0:2] = q-q1
            eq[2:4] = q-q2
            return eq
        
        # Derivative
        def dsplit_eq(y, dy, α2, dα2):
            t, dt   = y[0], dy[0]
            a1, da1 = y[1], dy[1]
            q, dq   = y[2:4], dy[2:4]
            a2, da2 = α2[0], dα2[0]
            (q1, dq1), _ = extremal(t0, q0, covector(q0, \
                                                     (a1, da1)), \
                                    (t, dt))
            (q2, dq2), _ = extremal(t0, q0, covector(q0, \
                                                     (a2, da2)), \
                                    (t, dt))
            eq, deq      = np.zeros(4), np.zeros(4)
            eq[0:2], deq[0:2] = q-q1, dq-dq1
            eq[2:4], deq[2:4] = q-q2, dq-dq2
            return eq, deq
        
        split_eq = nt.tools.tensorize(dsplit_eq, tvars=(1, 2), \
                                      full=True)(split_eq)

        # Function to compute the splitting locus
        def get_splitting_locus(q, a1, t, a2, α0, αf):
        
            # Options
            opt  = nt.path.Options(MaxStepSizeHomPar=0.05, \
                                   Display='off');
        
            # Initial solution
            y0 = np.array([t, a1, q[0], q[1]])
            b0 = a2
        
            # callback
            def progress_bis(infos):
                current   = b0-infos.pars[0]
                total     = αf-α0+b0-α0
                barLength = 50
                percent   = float(current * 100.0 / total)
                arrow = '-' * int(percent/100.0 * barLength - 1)+ \
                '>'
                spaces    = ' ' * (barLength - len(arrow))
        
                print('    Progress: [%s%s] %1.2f %%' % \
                      (arrow, spaces, round(percent, 2)), end='\r')
        
            # First homotopy
            print('\n > Get splitting locus\n')
            sol  = nt.path.solve(split_eq, y0, b0, α0, \
                                 options=opt, df=split_eq, \
                                 callback=progress_bis)
            ysol = sol.xf
        
            # callback
            def progress(infos):
                current   = b0-α0+infos.pars[0]-α0
                total     = αf-α0+b0-α0
                barLength = 50
                percent   = float(current * 100.0 / total)
                arrow = '-' * int(percent/100.0 * barLength - 1)+ \
                '>'
                spaces    = ' ' * (barLength - len(arrow))
        
                print('    Progress: [%s%s] %1.2f %%' % \
                      (arrow, spaces, round(percent, 2)), end='\r')
        
            # Splitting locus computation
            sol = nt.path.solve(split_eq, ysol, α0, αf, \
                                options=opt, df=split_eq, \
                                callback=progress)
            print('\n')
        
            return (sol.xout, sol.parsout)

        def get_splitting_locus_right(q, a1, t, a2, gap):
            α0  = 0*np.pi/2+gap
            αf  = 1*np.pi/2-gap
            sol = get_splitting_locus(q, a1, t, a2, α0, αf)
            return sol
        
        def get_splitting_locus_left(q, a1, t, a2, gap):
            α0  = 1*np.pi/2+gap
            αf  = 2*np.pi/2-gap
            sol = get_splitting_locus(q, a1, t, a2, α0, αf)
            return sol

        # Get splitting locus
        if data.contains('splitting_locus') and not restart:
            splitting_locus = data.get('splitting_locus')
            print('Splitting locus loaded')
        else:
            gap = 1e-3
            splitting_locus_left  = \
            get_splitting_locus_left(q_l, α2_l, tf_l, α1_l, gap)
            splitting_locus_right = \
            get_splitting_locus_right(q_r, α1_r, tf_r, α2_r, gap)
            splitting_locus = {
                "left": (splitting_locus_left[0].tolist(), \
                         splitting_locus_left[1].tolist()),
                "right": (splitting_locus_right[0].tolist(), \
                          splitting_locus_right[1].tolist())
            }
            data.update({'splitting_locus':splitting_locus})
            print('Splitting locus saved')
        
        def cut_time(α):
            α = α % (2*np.pi)
            if ((α >= 0*np.pi/2) and (α <= 1*np.pi/2)) or \
            ((α >= 3*np.pi/2) and (α <= 4*np.pi/2)): # right
                sp = splitting_locus["right"]
                ys = np.array(sp[0])
                ts = ys[:, 0]
                if ((α >= 0*np.pi/2) and (α <= 1*np.pi/2)): 
                    # homotopic parameter
                    αs     = np.array(sp[1])
                    αs_cut = ys[:, 1]
                else:
                    αs_cut = np.array(sp[1])
                    αs     = ys[:, 1]
            else: #left
                sp = splitting_locus["left"]
                ys = np.array(sp[0])
                ts = ys[:, 0]
                if ((α >= 1*np.pi/2) and (α <= 2*np.pi/2)): 
                    # homotopic parameter
                    αs     = np.array(sp[1])
                    αs_cut = ys[:, 1]
                else:
                    αs_cut = np.array(sp[1])
                    αs     = ys[:, 1]
            # get indice of α
            αs = αs % (2*np.pi)
            ind   = np.argmin(np.abs(α-αs))
            time  = ts[ind]
            q     = ys[ind, 2:4]
            α_cut = αs_cut[ind]
            y0    = np.zeros(4)
            y0    = [time, α_cut, q[0], q[1]]
            fun   = lambda y: split_eq(y, [α])
            dfun  = lambda y, dy: \
            split_eq((y, dy), ([α], [0.0]))[1]
            fun   = nt.tools.tensorize(dfun, tvars=(1,))(fun)
            opt   = nt.nle.Options(Display='off', MaxFEval=10, \
                                   TolX=1e-5)
            sol   = nt.nle.solve(fun, y0, df=fun, options=opt)
            time  = sol.x[0]
            return time

        self.cut_time = cut_time
                    
        # Function to plot the splitting locus
        splitting_plot = plottings.Splitting_Locus(splitting_locus)
        self.splitting_plot = splitting_plot

    def plot(self, fig=None, coords=plottings.Coords.SPHERE, geodesics=0, \
             conjugate=False, cut=False, wavefronts=False, tf=None, azimuth=140):

        if geodesics>0:
            if not(cut):
                cut_time_fun = None
            else:
                cut_time_fun = self.cut_time
            fig = self.geodesics_plot(coords=coords, fig=fig, nb_geodesics=geodesics, \
                               ratio_tf=1, cut=cut_time_fun, recompute=True, \
                               azimuth=azimuth, tf=tf)
            
        if conjugate:
            self.conjugate_plot(coords=coords, fig=fig, azimuth=azimuth)

        if cut:
            self.splitting_plot(coords=coords, fig=fig, azimuth=azimuth)

        if wavefronts:
            self.wavefronts_plot(coords=coords, fig=fig, azimuth=azimuth)

        return fig









    

