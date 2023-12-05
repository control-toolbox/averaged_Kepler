
        # --------- STOP HERE
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
            gap = 1e-3
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