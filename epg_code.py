import numpy as np
import matplotlib.pyplot as plt

# MR Physics with phase graphs 
# Tony Stöcker, 2025


def pg_plot(T, G=None, CPMG=False, LONGITUDINAL=True, PHASE_LABEL=True, STATE_LABELS=False,
                STATE_LABEL_NUMBERS=False, FA_LABELS=False, TIME_LABELS=False, TAU_LABELS=False, ECHOES=True, BLACK=False, 
                ECHO_COUNT=False, axis=False ):
    """
    phase graph plotting function

    Input
    T          : time intervals between RF pulses
    G          : slopes (gradients) between RF pulses
    ...        : several booleans to switch on or off labels or modify the plot in other ways
    ECHO_COUNT : if True, print the actual and maximum possible number of echoes for the sequence
    axis       : if not False, used as the plotting axis 

    Output
    axis       : the axis object of the plot
    FL         : object of flip-angle labels for later modification
    SL         : object of state labels for later modification
    TL         : object of time-interval labels for later modification

    Usage
    phase_graph([t1, t2, t3]) 
    Generates a phase graph for three pulses and subsequent dephasing intervals t1, t2, t3 
    (=> RF pulses at time points 0, t1, t2

    phase_graph([t1, t2, t3],[g1, g2, g3]) 
    same as before but with varying slopes [g1,g2,g3] for the dephasing intervals
    
    Examples
    phase_graph([1, 1, 2])
    phase_graph([1, 3, 6])
    phase_graph([1, 1, 1]), [1, 3, 5])
    phase_graph([1, 1, 1, 1],STATE_LABELS=True)
    """
    if G is None:
        G = np.ones(len(T))
    
    T = np.insert(T, 0, 0) # insert t=0 (origin) as time point of the first RF pulse
    CT = np.cumsum(T)
    
    P = np.array([0])      # this will be the large array of 3^len(T) states
    echoes = 0

    B='b'
    R='r'
    if BLACK:
        B='k'
        R='k'

    #plot in new figure or given axis
    if axis == False:
        fig, ax = plt.subplots()
    else:
        ax = axis

    # plot all states
    FL=[]
    SL=[]
    TL=[]
    for i in range(1, len(T)):
        NP = []
        for j in range(len(P)):
            if i>1 and CPMG and np.mod(P[j],2)==0:
                    continue
            # plot transverse dephasing states after the pulse in blue
            if P[j]*G[i-1] > 0:
                ax.plot(CT[i-1] + np.array([0, T[i]]), P[j] + np.array([0, G[i-1]]) * T[i], linewidth=1, color=B)
            else:
                TE = -P[j]/G[i-1] 
                # plot transverse rephasing states after the pulse in red
                ax.plot(CT[i-1] + np.array([0, min(TE,T[i])]), P[j] + np.array([0, G[i-1]]) * min(TE,T[i]), linewidth=1, color=R)
                if TE<T[i]:
                    #dephasing after echo again plotted in blue
                    ax.plot(CT[i-1] + np.array([TE,T[i]]), np.array([0, G[i-1]]) * (T[i]-TE), linewidth=1, color=B)
                    if TE>0:
                        echoes += 1
                        if ECHOES: ax.plot(CT[i-1] + TE, 0,'o'+R)
            #plot longitudinal states as black dashed lines
            if P[j] != 0 and LONGITUDINAL:
                ax.plot(CT[i-1] + np.array([0, T[i]]), P[j] + np.array([0, 0]), '--k', linewidth=1)

            if STATE_LABELS:
                SLN = ''
                if STATE_LABEL_NUMBERS: SLN='$^{(%d)}$' %(i) 
                N=len(P)//2
                p=G[0] * T[1]
                dist = 0.4
                if j==0:
                    SL.append(ax.text(CT[i-1] + 0.015*T[1],  dist*p, '$F_0$' + SLN, fontweight='bold', fontsize=8, color=B))
                    SL.append(ax.text(CT[i-1] + 0.025*T[1], -.7*dist*p, '$Z_0$' + SLN, fontweight='bold', fontsize=8, color='k'))
                if j>0 and j<=N:
                    SL.append(ax.text(CT[i-1] + 0.025*T[1], P[N+j]+dist*p, '$F_{%d}$' %int(P[N+j]) + SLN, fontweight='bold', fontsize=8, color=B))
                    SL.append(ax.text(CT[i-1] + 0.025*T[1], P[N+j]-.7*dist*p, '$Z_{%d}$' %int(P[N+j]) + SLN, fontweight='bold', fontsize=8, color='k'))
                    SL.append(ax.text(CT[i-1] + 0.025*T[1], P[N-j]+dist*p, '$F_{%d}$' %int(P[N-j]) + SLN, fontweight='bold', fontsize=8, color=R))
                    SL.append(ax.text(CT[i-1] + 0.025*T[1], P[N-j]-.7*dist*p, '$Z_{%d}$' %int(P[N-j]) + SLN, fontweight='bold', fontsize=8, color='k'))
            #state grow after pulse
            NP.extend([P[j], P[j] + G[i-1] * T[i], P[j] - G[i-1] * T[i]])

        #concatenate all states so far and remove double entries. np.unique sorts as well (which is needed!)
        P = np.unique(np.concatenate((P, NP))) 


    #plot x-axis (time) 
    ax.plot([0, CT[-1]], [0, 0], '-k', linewidth=2)
    y = ax.get_ylim()
    x = ax.get_xlim()

    #plot RF Pulses as vertical lines
    for i in range(len(T)-1):
        xp = CT[i]
        dx = CT[i+1] - CT[i]
        dy = y[1] - y[0]
        ax.plot([CT[i], CT[i]], y, 'k', linewidth=2)
        if TIME_LABELS:
            ax.text(CT[i] , y[0] - dy / 10, 't=%d' %CT[i], fontsize=10, horizontalalignment='center')
        if FA_LABELS:
            FL.append(ax.text(CT[i] , y[0] - dy / 10, '$\\alpha_{%d}$' %(i+1), fontsize=10, horizontalalignment='center'))
        if TAU_LABELS and i<len(T)-2:
            TL.append(ax.text(xp + dx / 2, y[1] - dy / 20, '$\\tau_{%d}$' %(i+1), fontsize=10, horizontalalignment='center'))
            ax.arrow(xp+.05*dx , y[1] - dy / 16,  .9*dx, 0, linewidth=1, color='k', length_includes_head=True,head_width=.02*dy, head_length=.01*dy)
            ax.arrow(xp+.95*dx , y[1] - dy / 16, -.9*dx, 0, linewidth=1, color='k', length_includes_head=True,head_width=.02*dy, head_length=.01*dy)
   
    #add $\theta$ label at the top of the y-axis (= first RF pulse)
    ax.axis('off')
    if PHASE_LABEL:
        ax.text(x[0], y[1] - (y[1] - y[0]) / 30, '$\\theta$', fontsize=10)
    
    if axis == False:
        plt.show()

    if ECHO_COUNT:
        n = len(T)-1
        print ('# echoes:',echoes, ' , max # echoes:', int((3**n-1)/4-n/2))
    
    return (ax,FL,SL,TL)

class basic_epg:
    def __init__(self, M0, T1, T2, TR):
        """
        simple EPG class for educational purposes.
        Most functionality is equivalent to the (much!) faster pyepg class https://github.com/mrphysics-bonn/EPGpp
        
        Parameters:
        M0 : Equilibrium magnetization
        T1 : Longitudinal relaxation time
        T2 : Transverse relaxation time
        TR : Repetition time of the sequence
        """
        self.step = 0
        self.M0 = M0
        self.T1 = T1
        self.T2 = T2
        self.TR = TR
        self.E1 = np.exp(-TR / T1)
        self.E2 = np.exp(-TR / T2)

        self.Fb = np.zeros((1,),dtype=complex)
        self.Zb = M0*np.ones((1,),dtype=complex)
        self.Fa = None
        self.Za = None

        self.phase = 0.0

    def Rotation(self, fa, ph):
        """
        rotation of states
        
        Parameters:
        fa : Flip angle (in radians)
        ph: Phase angle (in radians)        
        """
        ep = np.exp(1j*ph)
        c2 = np.cos(fa/2)**2
        s2 = np.sin(fa/2)**2
        c1 = np.cos(fa)
        s1 = np.sin(fa)
        
        # RF pulse transformation matrix
        TM = np.array([ [c2, s2*(ep**2), -1j*ep*s1], [s2/(ep**2), c2, (1j/ep)*s1], [-0.5*(1j/ep)*s1, 0.5*1j*ep*s1, c1] ])

        N = self.step  
        for i in range(N):
            RS = np.dot(TM, np.vstack((self.Fb[N-1+i], np.conj(self.Fb[N-1-i]), self.Zb[i])))
            self.Fa[N-1+i] = (RS[0])
            self.Fa[N-1-i] = np.conj(RS[1])
            self.Za[i]     = RS[2]

    def Step(self,fa,ph,RFSpoil=False):
        """
        Compute one step of the EPG => state grow, state rotation, and state relaxation 
        fa      : Flip angle (in degrees)
        ph     : Phase angle (in degrees)
        RFSpoil : if True, the phase angle is treated as an quadratic phase increment for RF spoiling
                  (i.e. multiplied with the current step number and added to the previous phase)
        """

        self.step += 1
        N = self.step  

        if RFSpoil:
            self.phase += self.step * ph
            self.phase = np.mod(self.phase, 360.0)
        else:
            self.phase = ph

        # Initialize arrays for next states
        self.Fa = np.zeros((2 * N - 1,),dtype=complex)
        self.Za = np.zeros((N,),dtype=complex)
        
        # rotation of states
        self.Rotation( np.deg2rad(fa), np.deg2rad(self.phase))

        # time evolution: extend state space and relaxation 
        # appending two zeros shifts the Fb state number by one !!
        self.Fb = np.append(self.Fa*self.E2, (0.0, 0.0)) 
        self.Zb = np.append(self.Za*self.E1, 0.0)
        self.Zb[0] += self.M0 * (1 - self.E1)

    def Steps(self,fa,ph,steps,RFSpoil=False):
        """
        Compute multiple steps with constant flip-angle and phase 
        fa      : Flip angle (in degrees)
        ph     : Phase angle (in degrees)
        steps   : number of steps
        RFSpoil : if True, the phase angle is treated as an quadratic phase increment for RF spoiling
                  (i.e. multiplied with the current step number and added to the previous phase)
        """
        for n in range(steps):
            self.Step(fa,ph,RFSpoil)
        return

    def Equilibrium(self):
        """
        set EPG to equilibrium
        """
        self.step = 0
        self.Fb   = np.zeros((1,),dtype=complex)
        self.Zb   = self.M0*np.ones((1,),dtype=complex)
        self.Fa   = None
        self.Za   = None
        self.phase = 0.0

    def SetParameters(self,M0,T1,T2,TR):
        """
        set new parameters for this EPG (resets to equilibrium as well)
        """
        self.M0 = M0
        self.T1 = T1
        self.T2 = T2
        self.TR = TR
        self.E1 = np.exp(-TR / T1)
        self.E2 = np.exp(-TR / T2)
        self.Equilibrium()
        return
    
    # Note: the following four getter-functions of complex magnetization are not available for the pyepg class
    def getFa(self,num):
        """
        get transverse magnetization after the last pulse
        Input num is the state order (e.g. num=0 => F_{0} state, num=-1 => F_{-1} state)
        """
        n = self.step-1-num
        return ( 0.0+1j*0.0 if (n<0 or n>=2*self.step-1) else self.Fa[n] )

    def getZa(self,num):
        """
        get longitudinal magnetization after the last pulse
        Input num is the state order (e.g. num=0 => F_{0} state, num=-1 => F_{-1} state)
        """
        return ( 0.0+1j*0.0 if (num<0 or num>=self.step) else self.Za[num] )

    def getFb(self,num):
        """
        get transverse magnetization before the next pulse
        Input num is the state order (e.g. num=0 => F_{0} state, num=-1 => F_{-1} state)
        """
        n = self.step-num
        return ( 0.0+1j*0.0 if (n<0 or n>=2*self.step-1) else self.Fb[n] )

    def getZb(self,num):
        """
        get longitudinal magnetization before the next pulse
        Input num is the state order (e.g. num=0 => F_{0} state, num=-1 => F_{-1} state)
        """
        return ( 0.0+1j*0.0 if (num<0 or num>=self.step) else self.Zb[num] )

    # for compatibility with the pyepg class: 
    # getter-functions for real and imaginary parts as well as magnitude of the magnetization
    def GetReFa(self,num=0): return np.real(self.getFa(num))
    def GetImFa(self,num=0): return np.imag(self.getFa(num))
    def GetReFb(self,num=0): return np.real(self.getFb(num))
    def GetImFb(self,num=0): return np.imag(self.getFb(num))

    def GetReZa(self,num=0): return np.real(self.getZa(num))
    def GetImZa(self,num=0): return np.imag(self.getZa(num))
    def GetReZb(self,num=0): return np.real(self.getZb(num))
    def GetImZb(self,num=0): return np.imag(self.getZb(num))

    def GetMagFa(self,num=0): return np.abs(self.getFa(num))
    def GetMagFb(self,num=0): return np.abs(self.getFb(num))
    def GetMagZa(self,num=0): return np.abs(self.getZa(num))
    def GetMagZb(self,num=0): return np.abs(self.getZb(num))
 
    def GetPhase(self): return self.phase   # returns the phase of the last pulse (might be needed for RF spoiled sequences)
    def GetStep(self) : return self.step    # returns the current number of steps 
    def GetTR(self)   : return self.TR      # returns repetition time 
    def GetT1(self)   : return self.T1      # returns longitudinal relaxation time 
    def GetT2(self)   : return self.T2      # returns transverse relaxation time 
    def GetM0(self)   : return self.M0      # returns equilibrium magnetization 

    def StepsToSS(self, fa, Qph=0, tol=0.0001, max_size=99999, verbose=False):
        """
        Compute number of pulses to reach a steady state
        Input
        fa   : constant flip angle
        Qph : quadratic phase increment for RF spoiling
        tol  : error tolerance for termination 
        returns the number of steps to reach steady state, or -1 if not converged
        """
        F, F_old = -1.0, -1.0
        ph = 0.0
        for i in range(max_size):
            ph += i * Qph
            ph = np.mod(ph, 360.0)
            self.Step(fa, ph)
            F = self.GetMagFa() 
            if np.abs((F - F_old) / self.M0) < tol:
                return i
            F_old = F
        if verbose:
            print("Warning: StepsToSS did not converge. Maximum number of possible states exceeded.")
        return -1

    def GetNextMagFa(self, fa, ph, num):
        """
        get magnitude transverse magnetization after the next pulse (without extending or changing the EPG!)
        Input
        fa   : flip angle (in degree)
        ph  : pulse phase (in degree)
        num  : EPG state order (num=0 => F_0 state)
        """
        self.step += 1 
        self.Fa = np.append(self.Fa, (0.0, 0.0)) 
        self.Za = np.append(self.Za, 0.0)
        self.Rotation( np.deg2rad(fa), np.deg2rad(ph))
        NextMagFa = self.GetMagFa(num)
        self.Rotation( np.deg2rad(-fa), np.deg2rad(ph))
        self.step -= 1 
        self.Fa = np.delete(self.Fa, [-2, -1]) 
        self.Za = np.delete(self.Za, [-1])
        return NextMagFa
