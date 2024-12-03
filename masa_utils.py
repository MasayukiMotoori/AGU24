import numpy as np
from scipy import optimize
from scipy.constants import mu_0, epsilon_0
from scipy import fftpack
from scipy import sparse
from scipy.special import factorial
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d, CubicSpline,splrep, BSpline
from scipy.sparse import csr_matrix, csc_matrix
import csv
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.linalg import lu_factor, lu_solve
from scipy import signal
import empymod
import discretize
import  os
eps= np.finfo(float).eps

class TikonovInversion:
    def __init__(self, G_f, Wd, Wx=None, m_ref=None,Prj_m=None,m_fix=None,sparse_matrix=False):  
        self.G_f = G_f
        self.Wd = Wd
        self.nD = G_f.shape[0]
        self.nP = G_f.shape[1]
        self.Wx = self.get_Wx()
        self.m_ref = m_ref
        self.Prj_m = Prj_m  
        self.m_fix = m_fix
        if Prj_m is not None:
            assert Prj_m.shape[0] == self.nP
            self.nM = Prj_m.shape[1]
        else:
            self.Proj_m = np.eye(self.nP)
            self.nM = self.nP
            self.m_fix = np.zeros(self.nP)
        self.sparse_matrix = sparse_matrix
    
    def get_Wx(self):
        nP = self.nP
        Wx = np.zeros((nP-1, nP))
        element = np.ones(nP-1)
        Wx[:,:-1] = np.diag(element)
        Wx[:,1:] += np.diag(-element)
        return Wx

    def recover_model(self, dobs, beta, sparse_matrix=False):
   
        # This is for the mapping 
        G_f = self.G_f
        Wd = self.Wd
        Wx = self.Wx
        Prj_m = self.Prj_m
        m_fix= self.m_fix
        sparse_matrix = self.sparse_matrix
        
        left =(Prj_m.T @G_f.T @ Wd.T @ Wd @ G_f@Prj_m
             + beta *Prj_m.T @Wx.T @ Wx@Prj_m)
        if sparse_matrix:
            left = csr_matrix(left)
        right = ( G_f.T @ Wd.T @Wd@ dobs@Prj_m
                 -m_fix.T@G_f.T@Wd.T@Wd@G_f@Prj_m
                 -beta* m_fix.T@Wx.T@Wx@Prj_m)
        m_rec = np.linalg.solve(left, right)
        #filt_curr = spsolve(left, right)
        rd = Wd@(G_f@Prj_m@m_rec-dobs)
        rm = Wx@Prj_m@m_rec

        phid = 0.5 * np.dot(rd, rd)
        phim = 0.5 * np.dot(rm,rm)
        p_rec = m_fix + Prj_m@m_rec
        return p_rec, phid, phim
    
    def tikonov_inversion(self,beta_values, dobs):
        n_beta = len(beta_values)
        nP= self.nP

        mrec_tik = np.zeros(nP, n_beta)  # np.nan * np.ones(shape)
        phid_tik = np.zeros(n_beta)
        phim_tik = np.zeros(n_beta) 
        for i, beta in enumerate(beta_values): 
            mrec_tik[:, i], phid_tik[i], phim_tik[i] = self.recover_model(
            dobs=dobs, beta=beta)
        return mrec_tik, phid_tik, phim_tik
    
    def estimate_beta_range(self, num=20, eig_tol=1e-12):
        G_f = self.G_f
        Wd = self.Wd
        Wx = self.Wx
        Proj_m = self.Prj_m  # Use `Proj_m` to map the model space

        # Effective data misfit term with projection matrix
        A_data = Proj_m.T @ G_f.T @ Wd.T @ Wd @ G_f @ Proj_m
        eig_data = np.linalg.eigvalsh(A_data)
        
        # Effective regularization term with projection matrix
        A_reg = Proj_m.T @ Wx.T @ Wx @ Proj_m
        eig_reg = np.linalg.eigvalsh(A_reg)
        
        # Ensure numerical stability (avoid dividing by zero)
        eig_data = eig_data[eig_data > eig_tol]
        eig_reg = eig_reg[eig_reg > eig_tol]

        # Use the ratio of eigenvalues to set beta range
        beta_min = np.min(eig_data) / np.max(eig_reg)
        beta_max = np.max(eig_data) / np.min(eig_reg)
        
        # Generate 20 logarithmically spaced beta values
        beta_values = np.logspace(np.log10(beta_min), np.log10(beta_max), num=num)
        return beta_values

class empymod_IPinv:

    def __init__(self, model_base, nlayer,
        m_ref=None, nD=0, nlayer_fix=0, Prj_m=None, m_fix=None,
        resmin=1e-3 , resmax=1e6, chgmin=1e-3, chgmax=0.9,
        taumin=1e-6, taumax=1e-1, cmin= 0.4, cmax=0.9,
        Wd = None, Ws=None, Wx=None, alphax=None, alphas=None,
        cut_off=None,filt_curr = None,  window_mat = None
        ):
        self.model_base = model_base
        self.nlayer = int(nlayer)
        self.nlayer_fix = int(nlayer_fix)
        self.nP = 4*(nlayer + nlayer_fix)
        self.m_ref = m_ref
        self.Prj_m = Prj_m  
        self.m_fix = m_fix
        if Prj_m is not None:
            assert Prj_m.shape[0] == self.nP
            self.nM = Prj_m.shape[1]
        else:
            self.Proj_m = np.eye(self.nP)
            self.nM = self.nP
            self.m_fix = np.zeros(self.nP)       
        self.nD = nD
        self.resmin = resmin
        self.resmax = resmax
        self.chgmin = chgmin
        self.chgmax = chgmax
        self.taumin = taumin
        self.taumax = taumax
        self.cmin = cmin
        self.cmax = cmax
        self.Wd = Wd
        self.Ws = Ws
        self.Wx = Wx
        self.alphax = alphax
        self.alphas = alphas
        self.cut_off = cut_off
        self.filt_curr = filt_curr
        self.window_mat = window_mat

    def get_param(self, param, default):
        return param if param is not None else default
        
    def fix_sea_basement(self, res_sea, res_base, 
                chg_sea, chg_base, tau_sea, tau_base, c_sea, c_base):
        ## return and set mapping for fixigin sea and basement resistivity
        ## Assert there are no fix ing at this stage
        nlayer = self.nlayer
        nlayer_fix=2
        nlayer_sum = nlayer+nlayer_fix
        Prj_m_A = np.block([
            [np.zeros(nlayer)], # sea water
            [np.eye(nlayer)], # layers
            [np.zeros(nlayer)], # basement
        ])
        Prj_m=np.block([
        [Prj_m_A, np.zeros((nlayer_sum, 3*nlayer))], # Resistivity
        [np.zeros((nlayer_sum,  nlayer)), Prj_m_A, np.zeros((nlayer_sum, 2*nlayer))], # Chargeability
        [np.zeros((nlayer_sum,2*nlayer)), Prj_m_A, np.zeros((nlayer_sum, nlayer))], # Time constant
        [np.zeros((nlayer_sum,3*nlayer)), Prj_m_A], # Exponent C
        ])
        m_fix = np.r_[ 
        np.log(res_sea), np.zeros(nlayer), np.log(res_base), # Resistivity
        chg_sea, np.zeros(nlayer), chg_base, # Chargeability
        np.log(tau_sea),np.zeros(nlayer), np.log(tau_base), # Time constant
        c_sea,np.zeros(nlayer),c_base # Exponent C
        ]
        assert len(m_fix) == 4*nlayer_sum
        self.nlayer_fix = nlayer_fix
        self.Prj_m = Prj_m
        self.m_fix = m_fix
        self.nP= Prj_m.shape[0]
        self.nM= Prj_m.shape[1]
        assert self.nP == 4*(nlayer+nlayer_fix)
        assert self.nM == 4*nlayer
        return Prj_m, m_fix

    def pelton_et_al(self, inp, p_dict):
        """ Pelton et al. (1978)."""

        # Compute complex resistivity from Pelton et al.
        iotc = np.outer(2j * np.pi * p_dict['freq'], inp['tau']) ** inp['c']
        rhoH = inp['rho_0'] * (1 - inp['m'] * (1 - 1 / (1 + iotc)))
        rhoV = rhoH * p_dict['aniso'] ** 2

        # Add electric permittivity contribution
        etaH = 1 / rhoH + 1j * p_dict['etaH'].imag
        etaV = 1 / rhoV + 1j * p_dict['etaV'].imag
        return etaH, etaV

    def get_ip_model(self, mvec):
        Prj_m = self.Prj_m
        m_fix = self.m_fix
        nlayer= self.nlayer
        nlayer_fix = self.nlayer_fix
        nlayer_sum = nlayer + nlayer_fix
        param = Prj_m @ mvec + m_fix
        res = np.exp(param[            :   nlayer_sum])
        m   =        param[  nlayer_sum: 2*nlayer_sum]
        tau = np.exp(param[2*nlayer_sum: 3*nlayer_sum])
        c   =        param[3*nlayer_sum: 4*nlayer_sum]
        pelton_model = {'res': res, 'rho_0': res, 'm': m,
                        'tau': tau, 'c': c, 'func_eta': self.pelton_et_al}
        return pelton_model


    def predicted_data(self, model_vector):
        cut_off = self.cut_off
        filt_curr = self.filt_curr
        window_mat = self.window_mat
        ip_model = self.get_ip_model(model_vector)
        data = empymod.bipole(res=ip_model, **self.model_base)
        self.nD = len(data)
        if cut_off is not None:
            times = self.model_base['freqtime']
            smp_freq = 1/(times[1]-times[0])
            data_LPF = self.apply_lowpass_filter(
                       data=data,cut_off=cut_off,smp_freq=smp_freq
                       )
            data = data_LPF
        if filt_curr is not None:
            data_curr = signal.convolve(data_LPF, filt_curr)[:len(data)]
            data = data_curr
        if window_mat is not None:
            data_window = window_mat @ data_curr
            self.nD = len(data_window)
            data = data_window
        return data
   
    def apply_lowpass_filter(self, data, cut_off,smp_freq, order=1):
        nyquist = 0.5 * smp_freq
        normal_cutoff = cut_off / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, data)
        return y

    def projection_halfspace(self, a, x, b):
        projected_x = x + a * ((b - np.dot(a, x)) / np.dot(a, a)) if np.dot(a, x) > b else x
        # Ensure scalar output if input x is scalar
        if np.isscalar(x):
            return float(projected_x)
        return projected_x

    def proj_c(self,mvec):
        "Project model vector to convex set defined by bound information"
        nlayer = self.nlayer
        a = np.r_[1]
        print(mvec)
        for j in range(nlayer):
            r_prj = mvec[j]
            m_prj = mvec[j+   nlayer]
            t_prj = mvec[j+ 2*nlayer]
            c_prj = mvec[j+ 3*nlayer]
            r_prj = float(self.projection_halfspace( a, r_prj,  np.log(self.resmax)))
            r_prj = float(self.projection_halfspace(-a, r_prj, -np.log(self.resmin)))
            m_prj = float(self.projection_halfspace( a, m_prj,  self.chgmax))
            m_prj = float(self.projection_halfspace(-a, m_prj, -self.chgmin))
            t_prj = float(self.projection_halfspace( a, t_prj,  np.log(self.taumax)))
            t_prj = float(self.projection_halfspace(-a, t_prj, -np.log(self.taumin)))
            c_prj = float(self.projection_halfspace( a, c_prj,  self.cmax))
            c_prj = float(self.projection_halfspace(-a, c_prj, -self.cmin))
            mvec[j         ] = r_prj
            mvec[j+  nlayer] = m_prj
            mvec[j+2*nlayer] = t_prj
            mvec[j+3*nlayer] = c_prj
        return mvec
  
    def clip_model(self, mvec):
        mvec_tmp = mvec.copy()
        nlayer = self.nlayer
        mvec_tmp[        :  nlayer]=np.clip(
            mvec[        :  nlayer], np.log(self.resmin), np.log(self.resmax)
            )
        mvec_tmp[  nlayer:2*nlayer]=np.clip(
            mvec[  nlayer:2*nlayer], self.chgmin, self.chgmax
            )
        mvec_tmp[2*nlayer:3*nlayer]=np.clip(
            mvec[2*nlayer:3*nlayer], np.log(self.taumin), np.log(self.taumax)
            )
        mvec_tmp[3*nlayer:4*nlayer]=np.clip(
            mvec[3*nlayer:4*nlayer], self.cmin, self.cmax
            )
        return mvec_tmp
    def Japprox(self, model_vector, perturbation=0.1, min_perturbation=1e-3):
        delta_m = min_perturbation  # np.max([perturbation*m.mean(), min_perturbation])
#        delta_m = perturbation  # np.max([perturbation*m.mean(), min_perturbation])
        J = []

        for i, entry in enumerate(model_vector):
            mpos = model_vector.copy()
            mpos[i] = entry + delta_m

            mneg = model_vector.copy()
            mneg[i] = entry - delta_m

            pos = self.predicted_data(mpos)
            neg = self.predicted_data(mneg)
            J.append((pos - neg) / (2. * delta_m))

        return np.vstack(J).T


    def get_Wd(self, dobs, dp=1, ratio=0.10, plateau=0):
        std = np.abs(dobs * ratio) ** dp + plateau
        Wd = np.diag(1 / std)
        self.Wd = Wd 
        return Wd

    def get_Ws(self):
        nlayer = self.nlayer
        nx = 4*nlayer
        Ws = np.diag(np.ones(nx))
        self.Ws = Ws
        return Ws

    def get_Wx(self):
        nlayer = self.nlayer
        if nlayer == 1:
            print("No smoothness for one layer model")
            Wx = np.zeros((4,4))
            self.Wx = Wx
            return Wx
        nx = nlayer - 1
        ny = nlayer
        Wx = np.zeros((4 * nx, 4 * ny))
        for i in range(4):
            Wx[i * nx:(i + 1) * nx, i * ny:(i + 1) * ny - 1] = -np.diag(np.ones(nx))
            Wx[i * nx:(i + 1) * nx, i * ny + 1:(i + 1) * ny] += np.diag(np.ones(nx))
        self.Wx = Wx
        return Wx

    def get_Wxx(self):

        e = np.ones(self.nlayers*4)

        p1 = np.ones(self.nlayers)
        p1[0] = 2
        p1[-1] = 0
        eup = np.tile(p1, 4)

        p2 = np.ones(self.nlayers)
        p2[0] = 0
        p2[-1] = 2
        edwn = np.tile(p2, 4)
        Wxx = np.diag(-2 * e) + np.diag(eup[:-1], 1) + np.diag(edwn[1:], -1)

        return Wxx

    def steepest_descent(self, dobs, model_init, niter):
        '''
        Eldad Haber, EOSC555, 2023, UBC-EOAS 
        '''
        model_vector = model_init
        r = dobs - self.predicted_data(model_vector)
        f = 0.5 * np.dot(r, r)

        error = np.zeros(niter + 1)
        error[0] = f
        model_itr = np.zeros((niter + 1, model_vector.shape[0]))
        model_itr[0, :] = model_vector

        print(f'Steepest Descent \n initial phid= {f:.3e} ')
        for i in range(niter):
            J = self.Japprox(model_vector)
            r = dobs - self.predicted_data(model_vector)
            dm = J.T @ r
            g = np.dot(J.T, r)
            Ag = J @ g
            alpha = np.mean(Ag * r) / np.mean(Ag * Ag)
            model_vector = self.constrain_model_vector(model_vector + alpha * dm)
            r = self.predicted_data(model_vector) - dobs
            f = 0.5 * np.dot(r, r)
            if np.linalg.norm(dm) < 1e-12:
                break
            error[i + 1] = f
            model_itr[i + 1, :] = model_vector
            print(f' i= {i:3d}, phid= {f:.3e} ')
        return model_vector, error, model_itr


    def Gradient_Descent(self, dobs, mvec_init, niter, beta, alphas, alphax,
            s0=1, sfac=0.5, stol=1e-6, gtol=1e-3, mu=1e-4, ELS=True, BLS=True ):
        """
        Perform the Gradient Descent algorithm for optimization.

        Parameters
        ----------
        dobs : ndarray
            The observed data.
        mvec_init : ndarray
            The initial model vector.
        niter : int
            The number of iterations to perform.
        beta : float
            The beta parameter for the algorithm.
        alphas : float
            The alpha_s parameter for the algorithm.
        alphax : float
            The alpha_x parameter for the algorithm.
        s0 : float, optional
            The initial step size (default is 1).
        sfac : float, optional
            The step size reduction factor (default is 0.5).
        stol : float, optional
            The step size tolerance (default is 1e-6).
        gtol : float
            The stopping criteria for the norm of the gradient.
        mu : float, optional
            The mu parameter for the algorithm (default is 1e-4).
        ELS : bool, optional
            Whether to use exact line search (default is True).
        BLS : bool, optional
            Whether to use backtracking line search (default is True).

        Returns
        -------
        mvec_new : ndarray
            The optimized model vector.
        error_prg : ndarray
            The progress of the error.
        mvec_prg : ndarray
            The progress of the model vector.

        """
        Wd = self.Wd
        Ws = self.Ws
        Wx = self.Wx

        mvec_old = mvec_init
        mvec_new = None
        mref = mvec_init
        error_prg = np.zeros(niter + 1)
        mvec_prg = np.zeros((niter + 1, mvec_init.shape[0]))
        rd = Wd @ (self.predicted_data(mvec_old) - dobs)
        phid = 0.5 * np.dot(rd, rd)
        rms = 0.5 * np.dot(Ws@(mvec_old - mref), Ws@(mvec_old - mref))
        rmx = 0.5 * np.dot(Wx @ mvec_old, Wx @ mvec_old)
        phim = alphas * rms + alphax * rmx
        f_old = phid + beta * phim
        k = 0
        error_prg[0] = f_old
        mvec_prg[0, :] = mvec_old
        print(f'Gradient Descent \n Initial phid = {phid:.2e} ,phim = {phim:.2e}, error= {f_old:.2e} ')
        for i in range(niter):
            # Calculate J:Jacobian and g:gradient
            J = self.Japprox(mvec_old)
            g = J.T @ Wd.T @ rd + beta * (alphas * Ws.T @ Ws @ (mvec_old - mref)
                                          + alphax * Wx.T @ Wx @ mvec_old)
            # Exact line search
            if ELS:
                t = np.dot(g,g)/np.dot(Wd@J@g,Wd@J@g)
#                t = (g.T@g)/(g.T@J.T@J@g)
            else:
                t = 1.

            # End inversion if gradient is smaller than tolerance
            g_norm = np.linalg.norm(g, ord=2)
            if g_norm < gtol:
                print(f"Inversion complete since norm of gradient is small as :{g_norm :.3e} ")
                break

            # Line search method Armijo using directional derivative
            s = s0
            dm = t*g
            directional_derivative = np.dot(g, -dm)

            mvec_new = self.proj_c(mvec_old - s * dm)
            rd = Wd @ (self.predicted_data(mvec_new) - dobs)
            phid = 0.5 * np.dot(rd, rd)
            rms = 0.5 * np.dot(Ws @ (mvec_new - mref), Ws @ (mvec_new - mref))
            rmx = 0.5 * np.dot(Wx @ mvec_new, Wx @ mvec_new)
            phim = alphas * rms + alphax * rmx
            f_new = phid + beta * phim
            if BLS:
                while f_new >= f_old + s * mu * directional_derivative:
                    s *= sfac
                    mvec_new = self.proj_c(mvec_old - s * dm)
                    rd = Wd @ (self.predicted_data(mvec_new) - dobs)
                    phid = 0.5 * np.dot(rd, rd)
                    rms = 0.5 * np.dot(Ws @ (mvec_new - mref), Ws @ (mvec_new - mref))
                    rmx = 0.5 * np.dot(Wx @ mvec_new, Wx @ mvec_new)
                    phim = alphas * rms + alphax * rmx
                    f_new = phid + beta * phim
                    if np.linalg.norm(s) < stol:
                        break
            mvec_old = mvec_new
            mvec_prg[i + 1, :] = mvec_new
            f_old = f_new
            error_prg[i + 1] = f_new
            k = i + 1
            print(f'{k:3}, s:{s:.2e}, gradient:{g_norm:.2e}, phid:{phid:.2e}, phim:{phim:.2e}, f:{f_new:.2e} ')
        # filter model prog data
        mvec_prg = mvec_prg[:k]
        error_prg = error_prg[:k]
        # Save Jacobian
        self.Jacobian = J
        return mvec_new, error_prg, mvec_prg

    def GaussNewton_smooth(self, dobs, mvec_init, niter,beta,
                           s0=1, sfac=0.5, stol=1e-6, gtol=1e-3, mu=1e-4):
        Wd = self.Wd
        Ws = self.Ws
        Wx = self.Wx
        alphas = self.alphas
        alphax = self.alphax
        mvec_old = mvec_init
        # applay initial mvec for reference mode
        mref = mvec_init
        # get noise part
        # Initialize object function
        rd = Wd @ (self.predicted_data(mvec_old) - dobs)
        phid = 0.5 * np.dot(rd, rd)
        rms = 0.5 * np.dot(mvec_old - mref, mvec_old - mref)
        rmx = 0.5 * np.dot(Wx @ mvec_old, Wx @ mvec_old)
        phim = alphas * rms + alphax * rmx
        f_old = phid + beta * phim
        # Prepare array for storing error and model in progress
        error_prg = np.zeros(niter + 1)
        mvec_prg = np.zeros((niter + 1, mvec_init.shape[0]))
        error_prg[0] = f_old
        mvec_prg[0, :] = mvec_old

        print(f'Gauss-Newton \n Initial phid = {phid:.2e} ,phim = {phim:.2e}, error= {f_old:.2e} ')
        for i in range(niter):

            # Jacobian
            J = self.Japprox(mvec_old)

            # gradient
            g = J.T @ Wd.T @ rd + beta * (alphas * Ws.T @ Ws @ (mvec_old - mref)
                                          + alphax * Wx.T @ Wx @ mvec_old)
            # Hessian approximation
            H = J.T @ Wd.T @ Wd @ J + beta * (alphas * Ws.T @ Ws + alphax * Wx.T @ Wx)

            # model step
            dm = np.linalg.solve(H, g)

            # End inversion if gradient is smaller than tolerance
            g_norm = np.linalg.norm(g, ord=2)
            if g_norm < gtol:
                print(f"Inversion complete since norm of gradient is small as :{g_norm :.3e} ")
                break

            # update object function
            s = s0
            mvec_new = self.clip_model(mvec_old - s * dm)
            rd = Wd @ (self.predicted_data(mvec_new) - dobs)
            phid = 0.5 * np.dot(rd, rd)
            rms = 0.5 * np.dot(mvec_new - mref, mvec_new - mref)
            rmx = 0.5 * np.dot(Wx @ mvec_new, Wx @ mvec_new)
            phim = alphas * rms + alphax * rmx
            f_new = phid + beta * phim

            # Backtracking method using directional derivative Amijo
            directional_derivative = np.dot(g, -dm)
            while f_new >= f_old + s * mu * directional_derivative:
                # backtracking
                s *= sfac
                # update object function
                mvec_new = self.clip_model(mvec_old - s * dm)
                rd = Wd @ (self.predicted_data(mvec_new) - dobs)
                phid = 0.5 * np.dot(rd, rd)
                rms = 0.5 * np.dot(Ws @ (mvec_new - mref), Ws @ (mvec_new - mref))
                rmx = 0.5 * np.dot(Wx @ mvec_new, Wx @ mvec_new)
                phim = alphas * rms + alphax * rmx
                f_new = phid + beta * phim
                # Stopping criteria for backtrackinng
                if s < stol:
                    break

            # Update model
            mvec_old = mvec_new
            mvec_prg[i + 1, :] = mvec_new
            f_old = f_new
            error_prg[i + 1] = f_new
            k = i + 1
            print(f'{k:3}, step:{s:.2e}, gradient:{g_norm:.2e}, phid:{phid:.2e}, phim:{phim:.2e}, f:{f_new:.2e} ')
        # clip progress of model and error in inversion
        error_prg = error_prg[:k]
        mvec_prg = mvec_prg[:k]

        return mvec_new, mvec_prg

    def objec_func(self,mvec,dobs,beta):
        Wd = self.Wd
        Ws = self.Ws
        Wx = self.Wx
        alphas = self.alphas
        alphax = self.alphax
        m_ref = self.m_ref
        rd = Wd @ (self.predicted_data(mvec) - dobs)
        phid = 0.5 * np.dot(rd, rd)
        rms = 0.5 * np.dot(Ws @ (mvec - m_ref), Ws @ (mvec - m_ref))
        rmx = 0.5 * np.dot(Wx @ mvec, Wx @ mvec)
        phim = alphas * rms + alphax * rmx
        f_obj = phid + beta * phim
        return f_obj, phid, phim

    def plot_model(self, model, ax, color='C0',linestyle='-', label="model", linewidth=1, depth_min=-100):
        depth = np.r_[depth_min+self.model_base["depth"][0], self.model_base["depth"]]
        depth_plot = np.vstack([depth, depth]).flatten(order="F")[1:]
        depth_plot = np.hstack([depth_plot, depth_plot[-1] * 1.5])
        model_plot = np.vstack([model, model]).flatten(order="F")
        ax.plot(model_plot, depth_plot,
             color=color, linestyle=linestyle, label=label,  linewidth=linewidth)
        return ax

    def plot_IP_par(self,mvec,color="orange", linestyle='-', label="",  linewidth=1.0,ax=None):
        if ax == None:
            fig, ax = plt.subplots(2, 2, figsize=(12, 8))

        # convert model vector to model
        model = self.get_ip_model(mvec)

    #    plot_model_m(model_base["depth"], model_ip["res"], ax[0], "resistivity","k")
        self.plot_model(model["res"], ax[0], color, linestyle=linestyle,label=label, linewidth=linewidth)

        self.plot_model(model["m"], ax[1], color, linestyle=linestyle,label=label, linewidth=linewidth)

        self.plot_model(model["tau"], ax[2],  color, linestyle=linestyle, label=label, linewidth=linewidth)

        self.plot_model(model["c"]  , ax[3],  color, linestyle=linestyle, label=label, linewidth=linewidth)

        ax[0].set_title("model_resistivity(ohm-m)")
        ax[1].set_title("model_changeability")
        ax[2].set_title("model_time_constant(sec)")
        ax[3].set_title("model_exponent_c")
        return ax

class TEM_Signal_Process:
    
    def __init__(self,  
        base_freq,on_time, rmp_time, rec_time, smp_freq,
        windows_cen=None, windows_strt = None, windows_end = None):
        self.base_freq = base_freq
        self.on_time = on_time
        self.rmp_time = rmp_time
        self.rec_time = rec_time
        self.smp_freq = smp_freq
        time_step = 1./smp_freq
        self.time_step = time_step
        self.times_rec = np.arange(0,rec_time,time_step) + time_step
        self.windows_cen= windows_cen
        self.windows_strt = windows_strt
        self.windows_end = windows_end
    
    def get_param(self, param, default):
        return param if param is not None else default

    def validate_times(self, times):
        if len(times) > 1:
            assert np.all(np.diff(times) >= 0), "Time values must be in ascending order."
    
    def get_param(self, param, default):
        return param if param is not None else default

    def get_windows_cen(self, windows_cen):
        self.validate_times( windows_cen)
        self.windows_cen = windows_cen
        windows_strt = np.zeros_like( windows_cen)
        windows_end = np.zeros_like( windows_cen)
        dt = np.diff( windows_cen)
        windows_strt[1:] =  windows_cen[:-1] + dt / 2
        windows_end[:-1] =  windows_cen[1:] - dt / 2
        windows_strt[0] =  windows_cen[0] - dt[0] / 2
        windows_end[-1] =  windows_cen[-1] + dt[-1] / 2
        self.windows_strt = windows_strt
        self.windows_end = windows_end
        return windows_strt,windows_end

    def get_window_linlog(self,linstep,time_trns):
        rmp_time = self.rmp_time
        rec_time = self.rec_time + rmp_time
        nlinstep = round(time_trns/linstep)
        logstep = np.log((linstep+time_trns)/time_trns)
        logstrt = np.log(time_trns)
#        logend = np.log(rec_time) + logstep + eps
        logend = np.log(rec_time) - eps
        nlogstep = round((logend-logstrt)/logstep)
        windows_cen= np.r_[np.arange(0,time_trns,linstep), np.exp(np.arange(logstrt,logend,logstep))]
        windows_strt = np.r_[np.arange(0,time_trns,linstep)-linstep/2, np.exp(np.arange(logstrt-logstep/2,logend-logstep/2,logstep))]
        windows_end =  np.r_[np.arange(0,time_trns,linstep)+linstep/2,  np.exp(np.arange(logstrt+logstep/2,logend+logstep/2,logstep))]
        self.windows_cen = windows_cen
        self.windows_strt = windows_strt
        self.windows_end = windows_end
        print(f'linear step: {nlinstep}, log step: {nlogstep}, total steps: {nlinstep+nlogstep}')
        return windows_cen, windows_strt, windows_end

    def get_window_log(self,logstep, tstart, tend=None, rmp_time=None):
        tend = self.get_param(tend, self.rec_time)
        rmp_time = self.get_param(rmp_time, self.rmp_time)
        if rmp_time is not None:
            tstart = tstart
            tend = tend - rmp_time     

        logstrt = np.log10(tstart)
        logend = np.log10(tend)
        log10_windows_cen = np.arange(logstrt,logend,logstep)
#        log10_windows_cen = np.linspace(logstrt,logend,logstep)
        windows_cen  = 10.**log10_windows_cen +rmp_time
        windows_strt = 10.**(log10_windows_cen-logstep/2) +rmp_time
        windows_end  = 10.**(log10_windows_cen+logstep/2) +rmp_time
        # windows_strt = 10.**(np.arange(logstrt-logstep/2,logend-logstep/2,logstep))
        # windows_end  = 10.**(np.arange(logstrt+logstep/2,logend+logstep/2,logstep))
        # print(f'log step: {len(self.windows_cen_log)}')
        self.windows_cen = windows_cen 
        self.windows_strt = windows_strt
        self.windows_end = windows_end
        return windows_cen, windows_strt, windows_end

    def window(self,times,data, windows_strt=None, windows_end=None):
        windows_strt = self.get_param(windows_strt, self.windows_strt)
        windows_end = self.get_param(windows_end, self.windows_end)
        self.validate_times(times)

        # Find bin indices for start and end of each windows
        start_indices = np.searchsorted(times, windows_strt, side='left')
        end_indices = np.searchsorted(times, windows_end, side='right')

        # Compute windowed averages
        data_window = np.zeros_like(windows_strt, dtype=float)
        for i, (start, end) in enumerate(zip(start_indices, end_indices)):
            if start < end:  # Ensure there are elements in the window
                data_window[i] = np.mean(data[start:end])
        return data_window
    
    def get_window_matrix (self, times, windows_strt=None, windows_end=None):
        windows_strt = self.get_param(windows_strt, self.windows_strt)
        windows_end = self.get_param(windows_end, self.windows_end)
        self.validate_times(times)
        nwindows = len(windows_strt)
        window_matrix = np.zeros((nwindows, len(times)))
        for i in range(nwindows):
            start = windows_strt[i]
            end = windows_end[i]
            ind_time = (times >= start) & (times <= end)
            if ind_time.sum() > 0:
                window_matrix[i, ind_time] = 1/ind_time.sum()
        return window_matrix

    def plot_window_data(self,data=None, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        windows_strt= self.windows_strt
        windows_end = self.windows_end
        windows_cen = self.windows_cen
        if data is None:
            ax.loglog(windows_cen, windows_cen,"k*")
            ax.loglog(windows_strt, windows_cen,"b|")
            ax.loglog(windows_end, windows_cen,"m|")
        else:
            assert len(data) == len(self.windows_cen), "Data and windows must have the same length."
            ax.loglog(windows_cen, data,"k*")
            ax.loglog(windows_strt, data,"b|")
            ax.loglog(windows_end, data,"m|")
        ax.grid(True, which="both")
        ax.legend(["center","start","end"])
        return ax

    def butter_lowpass(self, cutoff, order=1):
        fs = self.smp_freq
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a
   
    def apply_lowpass_filter(self, data, cutoff, order=1):
        b, a = self.butter_lowpass(cutoff, order=order)
        y = filtfilt(b, a, data)
        return y
    
    def filter_linear_rmp(self, rmp_time=None, times_rec=None, time_step=None):
        rmp_time  = self.get_param(rmp_time, self.rmp_time)
        times_rec = self.get_param(times_rec, self.times_rec)
        time_step = self.get_param(time_step, self.time_step)
        filter_linrmp = np.zeros_like(times_rec)
        inds_rmp = times_rec <= rmp_time
        filter_linrmp[inds_rmp] =   1.0/float(inds_rmp.sum())
        return filter_linrmp
    
    def filter_linear_rmp_rect(self, rmp_time=None):
        if rmp_time is None:
            rmp_time = self.rmp_time
        pos_off = self.filter_linear_rmp(rmp_time=rmp_time)
        return np.r_[-pos_off, pos_off]
        
    def rect_wave(self, t, base_freq=None, neg=False):
        self.get_param(base_freq, self.base_freq)
        if neg:
            pos= 0.5*(1.0+signal.square(2*np.pi*(base_freq*t    ),duty=0.25))
            neg=-0.5*(1.0+signal.square(2*np.pi*(base_freq*t+0.5),duty=0.25))
            return pos + neg
        else :
            pos= 0.5*(1.0+signal.square(2*np.pi*(base_freq*t    ),duty=0.5))
            return pos

    def rect_wave_rmp(self, t, base_freq=None, rmp_time=None,neg=False):
        self.get_param(base_freq, self.base_freq)
        self.get_param(rmp_time, self.rmp_time)
        if neg:
            print("under construction")
            return None
        else :
            pos= 0.5*(1.0+signal.square(2*np.pi*(base_freq*t    ),duty=0.5))
            ind_pos_on = t<=rmp_time
            pos[ind_pos_on] = t/rmp_time
            ind_pos_off = (t>=0.5/base_freq) & (t<=0.5/base_freq+rmp_time)
            pos[ind_pos_off] = 1.0 - (t-0.5/base_freq)/rmp_time
            return pos


    def interpolate_data(self,times,data, times_rec=None,method='linear',
        logmin_time=1e-8, linScale_time=1.0, logmin_data=1e-8, linScale_data=1.0):
        '''
        times (array-like): Original time points (not uniformly spaced).
        data (array-like): Original data values at time points `t`.
        method (str): Interpolation method ('linear', 'nearest', 'cubic', etc.).
        Returns:
            resampled_data (np.ndarray): Resampled data on `t_new`.
        '''
        times_rec = self.get_param(times_rec, self.times_rec)
        pslog_time = PsuedoLog(logmin=logmin_time, linScale=linScale_time)
        pslog_data = PsuedoLog(logmin=logmin_data, linScale=linScale_data)
        interpolator = interp1d(
            x=pslog_time.pl_value(times),
            y=pslog_data.pl_value(data),
            kind=method,
            fill_value='extrapolate'
        )
        
        return pslog_data.pl_to_linear(interpolator(pslog_time.pl_value(times_rec)))

    def deconvolve(self, data, data_pulse):
        filt, reminder = signal.deconvolve(
            np.r_[data, np.zeros(len(data)-1),
            data_pulse]
            )
        print(reminder)
        print(np.linalg.norm(reminder))
        return filt

class PsuedoLog:
    def __init__(self, logmin, linScale, max_y=eps, min_y=-eps,
        logminx=None, linScalex=None, max_x=eps, min_x=-eps):
        self.logmin = logmin
        self.linScale = linScale
        self.max_y = max_y
        self.min_y = min_y
        self.logminx = logminx
        self.linScalex = linScalex
        self.max_x = max_x
        self.min_x =min_x

    def get_param(self, param, default):
        return param if param is not None else default

    def pl_value(self, lin, logmin=None, linScale=None):    
        logmin = self.get_param(logmin, self.logmin)    
        linScale = self.get_param(linScale, self.linScale)
        # Check if `lin` is scalar
        is_scalar = np.isscalar(lin)
        if is_scalar:
            lin = np.array([lin])  # Convert scalar to array for uniform processing
                
        abs_lin = np.abs(lin)
        sign_lin = np.sign(lin)
        ind_pl = (abs_lin >= logmin)
        ind_lin = ~ind_pl
        plog = np.zeros_like(lin)
        plog[ind_pl] = sign_lin[ind_pl] * (
            np.log10(abs_lin[ind_pl] / logmin) + linScale
            )
        plog[ind_lin] = lin[ind_lin] / logmin * linScale
        return plog
    
    def pl_to_linear(self,plog, logmin=None, linScale=None):   
        logmin = self.get_param(logmin, self.logmin)    
        linScale = self.get_param(linScale, self.linScale)
        # Check if `lin` is scalar
        is_scalar = np.isscalar(plog)
        if is_scalar:
            lin = np.array([plog])  # Convert scalar to array for uniform processing
        abs_plog = np.abs(plog)
        sign_plog = np.sign(plog)
        ind_pl = (abs_plog >= linScale)
        ind_lin = ~ind_pl
        lin = np.zeros_like(plog)
        lin[ind_pl] = sign_plog[ind_pl] * logmin * 10 ** (abs_plog[ind_pl] - linScale)
        lin[ind_lin] = plog[ind_lin] / linScale * logmin
        return lin

    def semiply(self, x, y, logmin=None, linScale=None, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(1, 1)
            
        if len(x) > 1:
            assert np.all(np.diff(x) >= 0), "Time values must be in ascending order."
        
        logmin = self.get_param(logmin, self.logmin)
        linScale = self.get_param(linScale, self.linScale)
        plog_y = self.pl_value(lin=y, logmin=logmin, linScale=linScale)
        
        default_kwargs = {
            "linestyle": "-",
            "color": "orange",
            "linewidth": 1.0,
            "marker": None,
            "markersize": 1,
            "label": "pl_plot",
        }
        default_kwargs.update(kwargs)
        
        ax.semilogx(x, plog_y, **default_kwargs)
        
        self.max_y = max([self.max_y, max(y)])
        self.min_y = min([self.min_y, min(y)])
        return ax


    def semiplx(self, x, y,logminx=None,linScalex=None,ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        logminx = self.get_param(logminx, self.logminx)    
        linScalex = self.get_param(linScalex, self.linScalex)
        if len(x) > 1:
            assert np.all(np.diff(x) >= 0), "Time values must be in ascending order."
        plog_x = self.pl_value(lin=x, logmin=logminx, linScale=linScalex)

        default_kwargs = {
            "linestyle": "-",
            "color": "orange",
            "linewidth": 1.0,
            "marker": None,
            "markersize": 1,
            "label": "pl_plot",
        }
        default_kwargs.update(kwargs)
        
        ax.semilogx(plog_x, y, **default_kwargs)

        self.max_x = max([self.max_x,max(x)])
        self.min_x = min([self.min_x,min(x)])
        return ax

    def plpl_plot(self, x, y,
        logminx=None,linScalex=None,logmin=None,linScale=None,ax=None,**kwargs):
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        if len(x) > 1:
            assert np.all(np.diff(x) >= 0), "Time values must be in ascending order."
        logmin = self.get_param(logmin, self.logmin)    
        linScale = self.get_param(linScale, self.linScale)
        logminx = self.get_param(logminx, self.logminx)
        linScalex = self.get_param(linScalex, self.linScalex)
        plog_x = self.pl_value(lin=x, logmin=logminx, linScale=linScalex)
        plog_y = self.pl_value(lin=y, logmin=logmin, linScale=linScale)

        default_kwargs = {
            "linestyle": "-",
            "color": "orange",
            "linewidth": 1.0,
            "marker": None,
            "markersize": 1,
            "label": "pl_plot",
        }
        default_kwargs.update(kwargs)
        ax.plot(plog_x, plog_y, **default_kwargs)
        self.max_y = max([self.max_y,max(y)])
        self.min_y = min([self.min_y,min(y)])
        self.max_x = max([self.max_x,max(x)])
        self.min_x = min([self.min_x,min(x)])
        return ax

    def pl_axes(self,ax,logmin=None,linScale=None,max_y=None,min_y=None):
        logmin = self.get_param(logmin, self.logmin)    
        linScale = self.get_param(linScale, self.linScale)
        max_y = self.get_param(max_y, self.max_y)
        min_y= self.get_param(min_y, self.min_y)

        if max_y <= logmin:
            n_postick = 1
        else:
            n_postick= int(np.ceil(np.log10((max_y+eps)/logmin)+1.1))
        posticks = linScale + np.arange(n_postick)
        #poslabels = logmin*10**np.arange(n_postick)
        poslabels = [f"{v:.0e}" for v in (logmin * 10**np.arange(n_postick))]

        if -min_y <= logmin:
            n_negtick = 1
        else:
            n_negtick = int(np.ceil(np.log10((-min_y+eps)/logmin)+1.1))

        negticks = -linScale - np.arange(n_negtick)
        negticks = negticks[::-1]
        #neglabels = -logmin*10**np.arange(n_negtick)
        neglabels = [f"{v:.0e}" for v in (-logmin * 10**np.arange(n_negtick))[::-1]]
#        neglabels = neglabels[::-1]
#        ticks  = np.hstack(( negticks, [0], posticks))
        ticks  = np.r_[negticks, 0, posticks]
        labels = np.hstack((neglabels, [0], poslabels))
        ax.set_ylim([min(ticks), max(ticks)])
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)
        # reset max and min
        self.max_y = eps
        self.min_y = -eps
        return ax

    def pl_axes_x(self,ax,logminx=None,linScalex=None,max_x=None,min_x=None):
        logminx = self.get_param(logminx, self.logminx)    
        linScalex = self.get_param(linScalex, self.linScalex)
        max_x = self.get_param(max_x, self.max_x)
        min_x= self.get_param(min_x, self.min_x)
        if max_x <= logminx:
            n_postick = 1
        else:
            n_postick= int(np.ceil(np.log10(max_x/logminx)+1))
        posticks = linScalex + np.arange(n_postick)
        poslabels = [f"{v:.0e}" for v in (logminx * 10**np.arange(n_postick))]
        if -min_x <= logminx:
            n_negtick = 1
        else:
            n_negtick = int(np.ceil(np.log10(-min_x/logminx)+1))
        negticks = -linScalex - np.arange(n_negtick)
        negticks = negticks[::-1]
        neglabels = [f"{v:.0e}" for v in (-logminx * 10**np.arange(n_negtick))[::-1]]
        ticks  = np.r_[negticks, 0, posticks]
        labels = np.hstack((neglabels, [0], poslabels))
        ax.set_xlim([min(ticks), max(ticks)])
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
        # reset max and min
        self.max_x = eps
        self.min_x = -eps
        return ax

def solve_polynomial(a, n,pmax):
    # Coefficients of the polynomial -x^{n+1} + (1+a)x - a = 0
    coeffs = [-1] + [0] * (n-1) + [(1 + a), -a]  # [-1, 0, ..., 0, (1 + a), -a]
    
    # Find the roots of the polynomial
    roots = np.roots(coeffs)
    
    # Filter real roots
    real_roots = [r.real for r in roots if np.isreal(r)]
    
    # Find the real root closest to pmax
    if real_roots:
        closest_root = real_roots[np.argmin(np.abs(np.array(real_roots) - pmax))]
        return closest_root
    else:
        return None  # Return None if no real roots are found

def mesh_Pressure_Vessel(tx_radius,cs1,ncs1, pad1max,cs2,max,lim,pad2max): 
    h1a = discretize.utils.unpack_widths([(cs1, ncs1)])
    a1 = (tx_radius- np.sum(h1a))/cs1 
    n_tmp = -1 + np.log((a1+1)*pad1max-a1)/np.log(pad1max)
    npad1b= int(np.ceil(n_tmp))
    pad1 = solve_polynomial(a1, npad1b, pad1max)
    npad1c = int(np.floor(np.log(cs2/cs1)/np.log(pad1))-npad1b)
    if npad1c< 0:
        print("error: padx1max is too large")

    h1bc = discretize.utils.unpack_widths([(cs1, npad1b+npad1c, pad1)])

    ncs2 = int(np.ceil( (max-np.sum(np.r_[h1a,h1bc])) / cs2 ))

    h2a= discretize.utils.unpack_widths([(cs2, ncs2)])

    a2 = (lim-np.sum(np.r_[h1a, h1bc, h2a]))/cs2 
    n_tmp = -1 + np.log((a2+1)*pad2max-a2)/np.log(pad2max)
    npad2 = int(np.ceil(n_tmp))
    pad2 = solve_polynomial(a2, npad2, pad2max)
    h2b = discretize.utils.unpack_widths([(cs2, npad2, pad2)])
    h = np.r_[h1a,h1bc,h2a,h2b]
    return h




