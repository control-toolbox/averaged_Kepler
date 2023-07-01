import os.path

#import param
import panel as pn
css = '''
.bk.panel-widget-box {
  background: #f0f0f0;
  border-radius: 5px;
  border: 1px black solid;
}
'''
pn.extension('katex', 'mathjax', raw_css=[css])

import matplotlib.pyplot as plt           # for plots
from matplotlib.figure import Figure
from IPython.display import display, HTML
plt.rcParams.update({"text.usetex":True, "font.family":"sans-serif", "font.sans-serif":["Helvetica"]}) # font properties

from enum import Enum                     # for 2D vs 2D plots
from mpl_toolkits.mplot3d import Axes3D   # for 3D plots

import numpy as np
import nutopy as nt
import scipy as sc  # for integration with event detection

# ----------------------------------------------------------------------------------------------------
#
zc = 100
z_order_sphere     = zc; zc = zc+1
z_order_conj_surf  = zc; zc = zc+1
z_order_axes       = zc; zc = zc+1
z_order_geodesics  = zc; zc = zc+1
z_order_cut_locus  = zc; zc = zc+1
z_order_wavefront  = zc; zc = zc+1
z_order_conj_locus = zc; zc = zc+1
z_order_q0         = zc; zc = zc+1
delta_zo_back = 50

#
alpha_sphere = 0.4
alpha_conj   = 0.4

# Parameters for the 3D view
elev__ = -10
azimuth__ = 20
dist__ = 10

def get_cam(elev, azimuth, dist):
    ce   = np.cos(2*np.pi*elev/360)
    se   = np.sin(2*np.pi*elev/360)
    ca   = np.cos(2*np.pi*azimuth/360)
    sa   = np.sin(2*np.pi*azimuth/360)
    cam  = np.array([ dist*ca*ce, dist*sa*ce, dist*se])
    return cam

#u = cam / np.linalg.norm(cam)

# 2D to 3D coordinates
def coord3d(theta, phi, epsilon):
    v = theta
    u = phi
    coefs = (1., 1., epsilon)                   # Coefficients in (x/a)**2 + (y/b)**2 + (z/c)**2 = 1 
    rx, ry, rz = coefs                          # Radii corresponding to the coefficients
    x = rx * np.multiply(np.cos(u), np.cos(v))
    y = ry * np.multiply(np.cos(u), np.sin(v))
    z = rz * np.sin(u)
    return x, y, z

# Kind of coordinates
class Coords(Enum):
    CHART=2     # 2D
    SPHERE=3 # 3D

# plot surface from a closed curve defined by spherical coordinates
def surface_from_spherical_curve(x, y, epsilon):
    N = 100
    xmin = np.min(x)
    xmax = np.max(x)
    
    X = np.zeros((N, N))
    Y = np.zeros((N, N))
    
    xs = np.linspace(xmin, xmax, N)
    for i in range(N):
        x_current = xs[i]
        # find the two intersections of the curve with x_current
        ii  = np.argwhere(np.multiply(x[1:]-x_current, \
                                      x[0:-1]-x_current)<=0)
        #
        k   = ii[0][0]
        xk  = x[k]
        xkp = x[k+1]
        λ   = (x_current-xk)/(xkp-xk)
        y1  = y[k]+λ*(y[k+1]-y[k])
        #
        k   = ii[1][0]
        xk  = x[k]
        xkp = x[k+1]
        if abs(xkp-xk)>1e-12:
            λ = (x_current-xk)/(xkp-xk)
        else:
            λ = 0
        y2  = y[k]+λ*(y[k+1]-y[k])
        #
        ymin = min(y1, y2)
        ymax = max(y1, y2)
        ys = np.linspace(ymin, ymax, N)
        X[:, i] = x_current*np.ones(N)
        Y[:, i] = ys
    
    # cartesian
    XX = np.zeros((N, N))
    YY = np.zeros((N, N))
    ZZ = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            XX[i,j], YY[i,j], ZZ[i,j] = coord3d(X[i,j], Y[i,j], \
                                                epsilon)

    return XX, YY, ZZ

def decorate(ax, epsilon, coords, azimuth, q0):
    
    if(coords is Coords.CHART):

        # 2D
        x   = [-np.pi, np.pi, np.pi, -np.pi]
        y   = [np.pi/2, np.pi/2, np.pi/2+1, np.pi/2+1]
        ax.fill(x, y, color=(0.95, 0.95, 0.95))
        y   = [-(np.pi/2+1), -(np.pi/2+1), -np.pi/2, -np.pi/2]
        ax.fill(x, y, color=(0.95, 0.95, 0.95))
        ax.set_xlabel(r'$\theta$', fontsize=12)
        ax.set_ylabel(r'$\varphi$', fontsize=12)
        ax.axvline(0, color='k', linewidth=0.5, linestyle="dashed", zorder=z_order_axes)
        ax.axhline(0, color='k', linewidth=0.5, linestyle="dashed", zorder=z_order_axes)
        ax.axhline(-np.pi/2, color='k', linewidth=0.5, zorder=z_order_axes)
        ax.axhline( np.pi/2, color='k', linewidth=0.5, zorder=z_order_axes)
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(-np.pi/2, np.pi/2)
        ax.set_aspect('equal', 'box')

        if not(q0 is None):
            r = q0[0]
            θ = q0[1]
            φ = r - np.pi/2
            ax.plot(θ, φ, marker="o", markersize=3, 
                    markeredgecolor="black", markerfacecolor="black", zorder=z_order_q0)
        
    else:
    
        # 3D
        ax.set_axis_off();
        coefs = (1., 1., epsilon)              # Coefficients in (x/a)**2 + (y/b)**2 + (z/c)**2 = 1 
        rx, ry, rz = coefs                     # Radii corresponding to the coefficients

        # Set of all spherical angles:
        v = np.linspace(-np.pi, np.pi, 100)
        u = np.linspace(-np.pi/2, np.pi/2, 100)

        # Cartesian coordinates that correspond to the spherical angles
        x = rx * np.outer(np.cos(u), np.cos(v))
        y = ry * np.outer(np.cos(u), np.sin(v))
        z = rz * np.outer(np.sin(u), np.ones_like(v))

        # Plot:
        ax.plot_surface(x, y, z,  rstride=1, cstride=1, \
                        color=(0.99, 0.99, 0.99), alpha=alpha_sphere, \
                        antialiased=True, zorder=z_order_sphere)

        # initial point
        if not(q0 is None):
            r = q0[0]
            θ = q0[1]
            φ = r - np.pi/2
            x, y, z = coord3d(θ, φ, epsilon)
            cam = get_cam(elev__, azimuth, dist__)
            ps = x*cam[0]+y*cam[1]+z*cam[2]
            if ps>0: # back
                zo = z_order_q0 - delta_zo_back
                al = 0.5
            else:
                zo = z_order_q0
                al = 1.0
            ax.plot(x, y, z, marker="o", markersize=3, alpha=al, \
                    markeredgecolor="black", markerfacecolor="black", zorder=zo)

        # add one meridian
        N = 100
        if not(q0 is None):
            θ = q0[1]*np.ones(N)
        else:
            θ = 0*np.ones(N)
        φ = np.linspace(0, 2*np.pi, N)
        x, y, z = coord3d(θ, φ, epsilon)
        plot3d(ax, x, y, z, azimuth, color="black", \
               linewidth=0.5, linestyle="dashed", zorder=z_order_axes)

        # add equator
        N = 100
        θ = np.linspace(0, 2*np.pi, N)
        φ = 0*np.ones(N)
        x, y, z = coord3d(θ, φ, epsilon)
        plot3d(ax, x, y, z, azimuth, color="black", \
               linewidth=0.5, linestyle="dashed", zorder=z_order_axes)
        
        # Adjustment of the axes, so that they all have the same span:
        max_radius = max(rx, ry, rz)
        for axis in 'xyz':
            getattr(ax, 'set_{}lim'.format(axis))((-max_radius, max_radius))

        ax.view_init(elev=elev__, azim=azimuth) # Reproduce view
        #ax.dist = dist__
        ax.set_box_aspect(None, zoom=dist__)
        
        ax.set_xlim(np.array([-rx,rx])*.67)
        ax.set_ylim(np.array([-ry,ry])*.67)
        ax.set_zlim(np.array([-rz,rz])*.67)

        # 
        ax.set_aspect('equal', 'box')
    
# ----------------------------------------------------------------------------------------------------
# Initial plots
def plotInitFig(epsilon, coords, azimuth, q0=None):
    
    if(coords is Coords.CHART):

        # 2D
        fig = Figure(dpi=200)
        fig.set_figwidth(4)
        ax  = fig.add_subplot(111)
        
    else:
    
        # 3D
        fig = Figure(dpi=200)
        fig.set_figwidth(2)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        
        ax = fig.add_subplot(111, projection='3d')
        plt.tight_layout()

    decorate(ax, epsilon, coords, azimuth, q0)
        
    return fig

def plot3d(ax, x, y, z, azimuth, color, linewidth, linestyle='solid', zorder=1):
    N = len(x)
    i = 0
    j = 1
    cam = get_cam(elev__, azimuth, dist__)
    ps = x[0]*cam[0]+y[0]*cam[1]+z[0]*cam[2]
    while i<N-1:
        ps_j = x[j]*cam[0]+y[j]*cam[1]+z[j]*cam[2]
        if (ps*ps_j<0) or (j==N-1):
            if ps>0:
                ls = linestyle
                lw = linewidth/3.0
                al = 0.5
                zo = zorder - delta_zo_back
            else:
                ls = linestyle
                lw = linewidth
                al = 1.0
                zo = zorder
            ax.plot(x[i:j+1], y[i:j+1], z[i:j+1], color=color, \
                    linewidth=lw, linestyle=ls, zorder=zo, alpha=al)
            i = j
            ps = ps_j
        j = j+1
    
#----------------------------------------------------------------------------------------------------
# Get n indices of an array of size l
def indices(l, n):
    return range(0, l, l//n)

# Final time limit
tf_lim     = 5*np.pi
nb_geo_max = 4*48

class Geodesics:
    
    def __init__(self, geodesic, H, z0, extremal):
        self.fig         = None
        self.ax          = None
        self.geodesic    = geodesic
        self.extremal    = extremal
        self.Hamiltonian = H
        self.initial     = z0
        self.epsilon     = 1.0 # Sphere
        self.geodesics_saved = None
        self.embeded     = False
    
    # Main function: call __plots__ to plot geodescics
    def __call__(self, coords=Coords.CHART, fig=None, nb_geodesics=12, \
                 ratio_tf=1.0, cut=None, recompute=True, azimuth=azimuth__, tf=None):
        return self.__plots__(coords, fig=fig, nb_geodesics=nb_geodesics, \
                              ratio_tf=ratio_tf, cut=cut, recompute=recompute, azimuth=azimuth, tf=tf)
   
    # Function to plot a certain number of geodesics
    def __plots__(self, coords=Coords.CHART, fig=None, nb_geodesics=12, \
                  ratio_tf=1.0, cut=None, recompute=True, azimuth=azimuth__, tf=None):
        
        if ratio_tf>1.0 or ratio_tf<0:
            raise ArgumentValueError('ratio_tf must belong to [0, 1]')
        
        # check if the number of geodesics to plot is ok
        if nb_geodesics > nb_geo_max or nb_geodesics < 0:
            raise ArgumentValueError('nb_geodesics must belong to [0, ' + str(nb_geo_max) + ']')
        
        # if axfig is None, then we initialize the plot
        q0 = self.initial(0)[0:2]
        if fig is None:
            self.fig = plotInitFig(self.epsilon, coords, azimuth, q0)
        elif self.fig is None:
            self.fig = fig
        self.ax  = self.fig.get_axes()[0] 
        
        # we clear all before decorating: to restart the plots
        self.ax.clear()
        decorate(self.ax, self.epsilon, coords, azimuth, q0)
        
        if nb_geodesics == 0:
            return self.fig
              
        # Computation of the geodesics
        if recompute:
            self.geodesics_saved = {}
            αs1 = np.linspace(0*np.pi/2, 1*np.pi/2, (nb_geodesics//4)+1)
            αs2 = np.linspace(1*np.pi/2, 2*np.pi/2, (nb_geodesics//4)+1)
            αs3 = np.linspace(2*np.pi/2, 3*np.pi/2, (nb_geodesics//4)+1)
            αs4 = np.linspace(3*np.pi/2, 4*np.pi/2, (nb_geodesics//4)+1)
            αs = np.concatenate(np.array([αs1[0:-1], αs2[0:-1], αs3[0:-1], αs4[0:-1]]))
            list_state  = list([]); list_time   = list([]); list_α0 = list([])
            for α0 in αs:
                if (np.abs(α0 % (2*np.pi) - 0*np.pi/2) > 1e-8) and \
                (np.abs(α0 % (2*np.pi) - 1*np.pi/2) > 1e-8) and    \
                (np.abs(α0 % (2*np.pi) - 2*np.pi/2) > 1e-8) and    \
                (np.abs(α0 % (2*np.pi) - 3*np.pi/2) > 1e-8):
                    if tf is None:
                        if cut is None:
                            time = self.__get_final_time(α0)
                        else: # we plot until cut point
                            time = self.__get_cut_time__(α0, cut)
                    else:
                        time = tf
                    q = np.array(self.geodesic(time, α0))
                    list_state.append(q); list_time.append(time); list_α0.append(time)
                    self.geodesics_saved['state']  = list_state
                    self.geodesics_saved['time']   = list_time
                    self.geodesics_saved['α0'] = list_α0
        elif self.geodesics_saved is None:
            raise ArgumentValueError('You must compute the geodesics before. Set recompute=True.')

        list_state  = self.geodesics_saved['state']
        list_time   = self.geodesics_saved['time']
        list_α0     = self.geodesics_saved['α0']
        
        # Display the geodesics
        if recompute:
            indices_geo  = range(0, len(list_time))
        else:
            n_geo_total = len(list_time) 
            indices_geo = indices(n_geo_total, nb_geodesics)

        for i in indices_geo :
            q    = list_state[i]
            if cut is None:
                nnq = len(q[:,0])
                nnp = int(nnq * ratio_tf)
                p   = np.array([q[j, :] for j in range(0, nnp)]) # get the part to plot
                if nnp>0:
                    self.__plot__(p, coords, azimuth)
            else:
                self.__plot__(q, coords, azimuth)
        
        return self.fig

    # Interactive plot
    def interact(self, embed=False, restart=False, coords=Coords.CHART, azimuth=azimuth__):
        filename = 'geodesics.html'
        if not( embed and os.path.isfile(filename) ) or restart:
            pane = self.__plot_interactive__(embed=embed, filename=filename, \
                                             coords=coords, azimuth=azimuth)
        else:
            pane = None
        if embed:
            return HTML(filename)
        else:
            return pane
    
    # Interactive plot
    def __plot_interactive__(self, embed=False, filename='geodesics.html', \
                             coords=Coords.CHART, azimuth=azimuth__):

        #
        number_geodesics_slider_quart = 4
        ng         = number_geodesics_slider_quart*4
        ng2        = 2*ng
        #
        
        self.__plots__(coords=coords, nb_geodesics=ng2, ratio_tf=1.0, \
                       recompute=True, azimuth=azimuth);

        if(coords is Coords.CHART):
            v__ = 'Chart'
        else:
            v__ = 'Sphere'
        
        def myplot(ratio_tf=1.0, v=v__, n=5, azimuth=azimuth):
            if v=='Chart':
                c = Coords.CHART
            else:
                c = Coords.SPHERE
            return self.__plots__(coords=c, nb_geodesics=n, \
                                  ratio_tf=ratio_tf, recompute=False, azimuth=azimuth)

        ww = 130

        view_button = pn.widgets.RadioButtonGroup(name='View', value=v__, \
                                                  options=list(['Chart', 'Sphere']), \
                                                  button_type='primary', \
                                                  margin=(20, 10, 10, 10), width=ww) # H, D, B, G

        ratio_tf_slider  = pn.widgets.FloatSlider(name='Length', \
                                                  start=0.0, end=1.0, step=0.01, value=1.0, \
                                                  tooltips=False, margin=(10, 10, 10, 10), \
                                                  show_value=False, width=ww)

        number_geodesics_slider = pn.widgets.IntSlider(name='Quantity', \
                                                       value=ng, start=0, end=ng2, step=1, \
                                                       tooltips=False, margin=(10, 10, 10, 10), \
                                                       show_value=False, width=ww)

        azimuth_slider = pn.widgets.IntSlider(name='azimuth', \
                                              value=azimuth, start=0, end=360, step=10, \
                                              tooltips=False, margin=(10, 10, 10, 10), \
                                              show_value=False, width=ww)

        reactive_plot = pn.bind(myplot, ratio_tf_slider, view_button, \
                                number_geodesics_slider, azimuth_slider)
        
        pane = pn.Column( \
            pn.Row( \
                pn.Column(view_button, \
                          pn.Column(number_geodesics_slider, \
                                    ratio_tf_slider, \
                                    azimuth_slider), \
                          css_classes=['panel-widget-box'], margin=(50, 0, 0, 0)), \
                reactive_plot, margin=(-30, -30, 0, 0)) \
        ) #, legend)
        
        if embed:
            if not self.embeded:
                pane.save(filename, \
                          embed_states={ratio_tf_slider: \
                                        list(np.concatenate((np.arange(1, 0.7,-0.05), \
                                                             np.arange(0.7, 0, -0.1), [0.0]))), \
                                        number_geodesics_slider: [ng, ng2], \
                                        view_button: ('Chart', 'Sphere'), \
                                        azimuth_slider: [azimuth]}, \
                          progress=True, embed=True, title='Geodesics') #, #)
                                        #embed_json=True, json_prefix='geodesics_resources')           
                self.embeded = True
                
        return pane
    
    # Function to plot one given geodesic
    def __plot__(self, q, coords, azimuth):

        # from (r, θ) to (θ, φ), with φ = r - π / 2
        # r = π / 2 is the equator
        r = q[:, 0]
        θ = q[:, 1]
        φ = r - np.pi/2
        
        #
        if(coords is Coords.CHART):
            self.ax.plot(θ, φ, color='b', linewidth=0.5, zorder=z_order_geodesics)
        else:
            x, y, z = coord3d(θ, φ, self.epsilon)
            linewidth = 0.5
            plot3d(self.ax, x, y, z, azimuth, 'b', linewidth, zorder=z_order_geodesics)

    def __get_final_time(self, α0):

        #
        def Hvec(t, z):
            return self.Hamiltonian.vec(t, z[0:2], z[2:4])
        #
        # equator: r = π/2
        v0 = Hvec(0, self.initial(α0))
        s  = np.sign(v0[0])
        if s == 0:
            s = np.sign(v0[1])
        def hit_equator(t, z):
            if np.abs(α0 % (2*np.pi) - np.pi/2) <= 1e-8: # going up
                return z[0] - (np.pi-1e-3) # we stop before north/south pole
            elif np.abs(α0 % (2*np.pi) - 3*np.pi/2) <= 1e-8: # going down
                return z[0] - (1e-3) # we stop before north/south pole
            elif np.abs(α0 % (2*np.pi) - 0) <= 1e-8: # going right
                return z[1] - np.pi # half turn
            elif np.abs(α0 % (2*np.pi) - np.pi) <= 1e-8: # going left
                return -(z[1] - (-np.pi)) # half turn
            else:
                if t > 0:
                    return z[0] - np.pi/2 
                else:
                    return s
    
        hit_equator.terminal  = True
        hit_equator.direction = s

        z0 = self.initial(α0)
        sol = sc.integrate.solve_ivp(Hvec, [0, 5.0*np.pi], z0, events=hit_equator, dense_output=True)

        #
        if len(sol.t_events[0]) > 0:
            time = sol.t_events[0][0]
            # improve accuracy of time
            ti = 0.9*time
            qi, pi = self.extremal(0, z0[0:2], z0[2:4], ti)
            def myfun(t):
                qf, pf = self.extremal(ti, qi, pi, t)
                return hit_equator(t, qf)
            root = sc.optimize.brentq(myfun, 0.9*time, 1.1*time)
            time = root
        else:
            time = sol.t[-1]

        return time

    def __get_cut_time__(self, α0, cut):
        return cut(α0)
    
class Error(Exception):
    """
        This exception is the generic class
    """
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class ArgumentValueError(Error):
    """
        This exception may be raised when one argument of a function has a wrong value
    """
    
# ----------------------------------------------------------------------------------------------------
class Conjugate_Locus:
    
    def __init__(self, conjugate_locus):
        self.conjugate_locus = conjugate_locus
        self.epsilon = 1.0
    
    # Function to plot the conjugate locus
    def __call__(self, coords=Coords.CHART, fig=None, azimuth=azimuth__):
        # if fig is None, then we initialize the plot
        if fig is None:
            fig = plotInitFig(self.epsilon, coords, azimuth)
        ax  = fig.get_axes()[0]
        #
        c_l = self.conjugate_locus["left"]
        c_r = self.conjugate_locus["right"]
        #
        if coords is Coords.SPHERE:
            azimuth = ax.azim
            rs = np.concatenate([c_r[:, 1], c_l[:, 1]]) % (2*np.pi)
            θs = np.concatenate([c_r[:, 2], c_l[:, 2]]) % (2*np.pi)
            # plot the curve
            self.__plot_conj(rs, θs, coords, ax, azimuth)
            # plot the surface
            φs = rs-np.pi/2
            X, Y, Z = surface_from_spherical_curve(θs, φs, \
                                                   self.epsilon)
            ax.plot_surface(X, Y, Z,  rstride=1, cstride=1, \
                            color='r', alpha=alpha_conj, antialiased=True, \
                            zorder=z_order_conj_surf)
        else:
            # we plot separately
            self.__plot_conj(c_l[:, 1], c_l[:, 2], coords, ax, azimuth)
            self.__plot_conj(c_r[:, 1], c_r[:, 2], coords, ax, azimuth)   
        #
        return fig

    def __plot_conj(self, r, θ, coords, ax, azimuth):
        #r = q[:,0] #y[:,1]
        #θ = q[:,1] #y[:,2]
        φ = r - np.pi/2
        if(coords is Coords.CHART):
            ax.plot(θ, φ, color='r', linewidth=1.0, zorder=z_order_conj_locus)
        else:    
            x, y, z = coord3d(θ, φ, self.epsilon)
            linewidth = 1.0
            plot3d(ax, x, y, z, azimuth, 'r', linewidth, zorder=z_order_conj_locus)
        
# ----------------------------------------------------------------------------------------------------
class Wavefronts:
    
    def __init__(self, wavefronts):
        self.wavefronts = wavefronts
        self.epsilon    = 1.0
    
    # Function to plot the wavefront
    def __call__(self, coords=Coords.CHART, fig=None, azimuth=azimuth__):
        # if fig is None, then we initialize the plot
        if fig is None:
            fig = plotInitFig(self.epsilon, coords, azimuth)
        ax  = fig.get_axes()[0] 
        #
        if coords is Coords.SPHERE:
            azimuth = ax.azim
        #
        for w in self.wavefronts:
            w_l = w["left"]
            w_r = w["right"]
            wa_l = np.array(w_l[0])
            wa_r = np.array(w_r[0])
            if coords is Coords.SPHERE:
                rs = np.concatenate([wa_r[:, 0], wa_l[:, 0]]) % (2*np.pi)
                θs = np.concatenate([wa_r[:, 1], wa_l[:, 1]]) % (2*np.pi)
                self.__plot_wavefront(rs, θs, coords, ax, azimuth)
            else:
                self.__plot_wavefront(wa_l[:, 0], wa_l[:, 1], coords, ax, azimuth)
                self.__plot_wavefront(wa_r[:, 0], wa_r[:, 1], coords, ax, azimuth)
        #
        return fig

    def __plot_wavefront(self, r, θ, coords, ax, azimuth):
        #r = wa[:, 0]
        #θ = wa[:, 1]
        φ = r - np.pi/2
        if(coords is Coords.CHART):
            ax.plot(θ, φ, color='g', linewidth=1.0, zorder=z_order_wavefront)
        else:
            x, y, z = coord3d(θ, φ, self.epsilon)
            linewidth = 1.0
            plot3d(ax, x, y, z, azimuth, 'g', linewidth, zorder=z_order_wavefront)

# ----------------------------------------------------------------------------------------------------
class Splitting_Locus:
    
    def __init__(self, splitting_locus):
        self.splitting_locus = splitting_locus
        self.epsilon = 1.0
    
    # Function to plot the conjugate locus
    def __call__(self, coords=Coords.CHART, fig=None, azimuth=azimuth__):
        # if fig is None, then we initialize the plot
        if fig is None:
            fig = plotInitFig(self.epsilon, coords, azimuth)
        ax  = fig.get_axes()[0] 
        #
        #
        s_l = np.array(self.splitting_locus["left"][0])
        s_r = np.array(self.splitting_locus["right"][0])
        #
        if coords is Coords.SPHERE:
            azimuth = ax.azim
            rs = np.concatenate([s_r[:, 2], s_l[:, 2]]) % (2*np.pi)
            θs = np.concatenate([s_r[:, 3], s_l[:, 3]]) % (2*np.pi)
            self.__plot_splitting(rs, θs, coords, ax, azimuth)
        else:
            self.__plot_splitting(s_l[:, 2], s_l[:, 3], coords, ax, azimuth)
            self.__plot_splitting(s_r[:, 2], s_r[:, 3], coords, ax, azimuth)
        
        return fig

    def __plot_splitting(self, r, θ, coords, ax, azimuth):
        #r = split[:, 2]
        #θ = split[:, 3]
        φ = r - np.pi/2
        if(coords is Coords.CHART):
            ax.plot(θ, φ, color='k', linewidth=1.0, zorder=z_order_cut_locus)
        else:
            x, y, z = coord3d(θ, φ, self.epsilon)
            linewidth = 1.0
            plot3d(ax, x, y, z, azimuth, 'k', linewidth, zorder=z_order_cut_locus)



