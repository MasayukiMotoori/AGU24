import numpy as np
from scipy import fftpack
#from scipy.special import gamma

eps = np.finfo(float).eps

class InducedPolarization:

    def __init__(self,
        res0=None, con8=None, eta=None, tau=None, c=None,
        freq=None, times=None, windows_strt=None, windows_end=None
        ):

        if res0 is not None and con8 is not None and eta is not None:
            assert np.allclose(con8 * res0 * (1 - eta), 1.)
        self.con8 = con8
        self.res0 = res0
        self.eta = eta
        if self.res0 is None and self.con8 is not None and self.eta is not None:
            self.res0 = 1./ (self.con8 * (1. - self.eta))
        if self.res0 is not None and self.con8 is None and self.eta is not None:
            self.con8 = 1./ (self.res0 * (1. - self.eta))
        self.tau = tau
        self.c = c
        self.freq = freq
        self.times = times
        self.windows_strt = windows_strt
        self.windows_end = windows_end

    def validate_times(self, times):
        assert np.all(times >= -eps ), "All time values must be non-negative."
        if len(times) > 1:
            assert np.all(np.diff(times) >= 0), "Time values must be in ascending order."
    
    def get_param(self, param, default):
        return param if param is not None else default

    def pelton_res_f(self, freq=None, res0=None, eta=None, tau=None, c=None):
        freq = self.get_param(freq, self.freq)
        res0 = self.get_param(res0, self.res0)
        eta = self.get_param(eta, self.eta)
        tau = self.get_param(tau, self.tau)
        c = self.get_param(c, self.c)
        iwtc = (1.j * 2. * np.pi * freq*tau) ** c
        return res0*(1.-eta*(1.-1./(1. + iwtc)))

    def pelton_con_f(self, freq=None, con8=None, eta=None, tau=None, c=None):
        freq = self.get_param(freq, self.freq)
        con8 = self.get_param(con8, self.res0)
        eta = self.get_param(eta, self.eta)
        tau = self.get_param(tau, self.tau)
        c = self.get_param(c, self.c)
        iwtc = (1.j * 2. * np.pi * freq*tau) ** c
        return con8-con8*(eta/(1.+(1.-eta)*iwtc))

    def debye_con_t(self, times=None, con8=None, eta=None, tau=None):
        times = self.get_param(times, self.times)
        con8 = self.get_param(con8, self.res0)
        eta = self.get_param(eta, self.eta)
        tau = self.get_param(tau, self.tau)
        self.validate_times(times)            
        debye = np.zeros_like(times)
        ind_0 = (times == 0)
        debye[ind_0] = 1.0
        debye[~ind_0] = -eta/((1.0-eta)*tau)*np.exp(-times[~ind_0]/((1.0-eta)*tau))
        return con8*debye

    def debye_con_t_intg(self, times=None, con8=None, eta=None, tau=None):
        times = self.get_param(times, self.times)
        con8 = self.get_param(con8, self.res0)
        eta = self.get_param(eta, self.eta)
        tau = self.get_param(tau, self.tau)
        self.validate_times(times)            
        return con8 *(1.0 -eta*(1. -np.exp(-times/((1.0-eta)*tau))))

    def debye_res_t(self, times=None, res0=None, eta=None, tau=None):
        times = self.get_param(times, self.times)
        res0 = self.get_param(res0, self.res0)
        eta = self.get_param(eta, self.eta)
        tau = self.get_param(tau, self.tau)
        self.validate_times(times)            
        debye = np.zeros_like(times)
        res8 = res0 * (1.0 - eta)
        ind_0 = (times == 0)
        debye[ind_0] = res8 
        debye[~ind_0] = (res0-res8)/tau * np.exp(-times[~ind_0] / tau)
        return debye

    def debye_res_t_intg(self, times=None, res0=None, eta=None, tau=None):
        times = self.get_param(times, self.times)
        res0 = self.get_param(res0, self.res0)
        eta = self.get_param(eta, self.eta)
        tau = self.get_param(tau, self.tau)
        self.validate_times(times)            
        res8 = res0 * (1.0 - eta)
        return res8 + (res8 - res0)*(np.exp(-times/tau) - 1.0)

    def freq_symmetric(self,f):
        symmetric = np.zeros_like(f, dtype=complex)
        nstep = len(f)
        half_step = nstep // 2
        symmetric[:half_step] = f[:half_step]
        symmetric[half_step:] = f[:half_step].conj()[::-1]
        assert np.allclose(symmetric[:half_step].real, symmetric[half_step:].real[::-1])
        assert np.allclose(symmetric[:half_step].imag, -symmetric[half_step:].imag[::-1])
        return symmetric

    def get_frequency_tau(self, tau=None, log2nfreq=16): 
        tau = self.get_param(tau, self.tau)
        log2nfreq = int(log2nfreq)
        nfreq = 2**log2nfreq
        freqcen = 1 / tau
        freqend = freqcen * nfreq**0.5
        freqstep = freqend / nfreq
        freq = np.arange(0, freqend, freqstep)
        self.freq = freq
        print(f'log2(len(freq)) {np.log2(len(freq))} considering tau')
        return freq

    def get_frequency_tau2(self, tau=None, log2min=-8, log2max=8):
        tau = self.get_param(tau, self.tau)
        freqcen = 1 / tau
        freqend = freqcen * 2**log2max
        freqstep = freqcen * 2**log2min
        freq = np.arange(0, freqend, freqstep)
        self.freq = freq
        print(f'log2(len(freq)) {np.log2(len(freq))} considering tau')
        return freq


    def get_frequency_tau_times(self, tau=None, times=None,log2min=-8, log2max=8):
        tau = self.get_param(tau, self.tau)
        times = self.get_param(times, self.times)
        self.validate_times(times)
        _, windows_end = self.get_windows(times)

        freqstep = 1/tau*(2**np.floor(np.min(
            np.r_[log2min,np.log2(tau/windows_end[-1])]
        )))
        freqend = 1/tau*(2**np.ceil(np.max(
            np.r_[log2max, np.log2(2*tau/min(np.diff(times)))]
        )))
        freq = np.arange(0,freqend,freqstep)
        self.freq=freq
        print(f'log2(freq) {np.log2(len(freq))} considering tau and times')
        return freq

    def compute_fft(self, fft_f, freqend, freqstep):
        fft_f = self.freq_symmetric(fft_f)
        fft_data = fftpack.ifft(fft_f).real * freqend
        fft_times = np.fft.fftfreq(len(fft_data), d=freqstep)
        return fft_times[fft_times >= 0], fft_data[fft_times >= 0]

    def pelton_fft(self, con_form=True, con8=None, res0=None, eta=None, tau=None, c=None, freq=None):
        res0 = self.get_param(res0, self.res0)
        eta = self.get_param(eta, self.eta)
        tau = self.get_param(tau, self.tau)
        c = self.get_param(c, self.c) 
        freq = self.get_param(freq, self.freq) 
        freqstep = freq[1] - freq[0]
        freqend = freq[-1] +freqstep

        if con_form:
            con8 = self.get_param(con8, self.con8)
            fft_f = self.pelton_con_f(freq=freq,
                     con8=con8, eta=eta, tau=tau, c=c)
        else:
            res0 = self.get_param(res0, self.res0)
            fft_f = self.pelton_res_f(freq=freq,
                     res0=res0, eta=eta, tau=tau, c=c)
        fft_times, fft_data = self.compute_fft(fft_f, freqend, freqstep)
        return fft_times, fft_data

    def get_windows(self, times):
        self.validate_times(times)
        windows_strt = np.zeros_like(times)
        windows_end = np.zeros_like(times)
        dt = np.diff(times)
        windows_strt[1:] = times[:-1] + dt / 2
        windows_end[:-1] = times[1:] - dt / 2
        windows_strt[0] = times[0] - dt[0] / 2
        windows_end[-1] = times[-1] + dt[-1] / 2
        self.windows_strt = windows_strt
        self.windows_end = windows_end
        return windows_strt,windows_end

    def apply_windows(self, times, data, windows_strt=None, windows_end=None):
        if windows_strt is None:
            windows_strt = self.windows_strt
        if windows_end is None:
            windows_end = self.windows_end
        self.validate_times(times)

        # Find bin indices for start and end of each window
        start_indices = np.searchsorted(times, windows_strt, side='left')
        end_indices = np.searchsorted(times, windows_end, side='right')

        # Compute windowed averages
        window_data = np.zeros_like(windows_strt, dtype=float)
        for i, (start, end) in enumerate(zip(start_indices, end_indices)):
            if start < end:  # Ensure there are elements in the window
                window_data[i] = np.mean(data[start:end])

        return window_data


    # def freq_from_time(self,times=None, tau=None,log2min=-8, log2max=8):
    #     '''
    #     return frequencies given time 
    #     range is widered than 2**16 centered at 1/tau 
    #     '''
    #     tau = self.get_param(tau, self.tau)
    #     times = self.get_param(times, self.times)
    #     self.validate_times(times)

    #     windows_strt, windows_end = self.get_windows(times)

    #     freqstep = 1/tau*(2**np.floor(np.min(
    #         np.r_[log2min,np.log2(tau/windows_end[-1])]
    #     )))
    #     freqend = 1/tau*(2**np.ceil(np.max(
    #         np.r_[log2max, np.log2(2*tau/min(np.diff(times)))]
    #     )))
    #     freq = np.arange(0,freqend,freqstep)
    #     self.freq=freq
    #     print(f'increased log2(free){np.log2(len(freq))}')
    #     return freq    

    # def fft_t(self,times=None,con_form=True
    #     ,con8=None, res0=None, eta=None, tau=None,c=None
    #     ,filter=None):
    #     '''
    #     return the time domain pelton given times
    #     window is create given times
    #     frequency is considered given time 
    #     frequence range is widered atleast to 2**16 centered at 1/tau 
    #     '''
    #     if con_form :
    #         con8 = self.get_param(con8, self.con8)
    #     else:
    #         res0 = self.get_param(res0, self.res0)
    #     eta = self.get_param(eta, self.eta)
    #     tau = self.get_param(tau, self.tau)
    #     c = self.get_param(c, self.c)
 
    #     fft_times, fft_data = self.pelton_fft(
    #         con_form=con_form,con8=con8, res0=res0
    #         ,eta=eta, tau=tau,c=c, freq=freq)
    #     fft_time_step = fft_times[1] - fft_times[0]
    #     fft_data_intg = np.cumsum(fft_data) * fft_time_step

    #     if filter is not None:
    #         fft_data = np.convolve(fft_data,filter,mode='same')

    #     windowed_data = self.apply_windows(
    #     times=fft_times, data= fft_data)

    #     windowed_data_intg = self.apply_windows(
    #     times=fft_times, data= fft_data_intg)

    #     return windowed_data, windowed_data_intg

    # def pelton_con_t_fft(self, con8=None, eta=None, tau=None,c=None,
    #     freq= None):
    #     con8 = self.get_param(con8, self.res0)
    #     eta = self.get_param(eta, self.eta)
    #     tau = self.get_param(tau, self.tau)
    #     c = self.get_param(c, self.c)   
    #     if freq is None:
    #         freq= self.generate_frequency(tau)
    #     self.freq = freq
    #     freqend = freq[-1]
    #     freqstep = freq[1] - freq[0]
    #     fft_f = self.pelton_con_f(freq=freq, con8=con8, eta=eta, tau=tau, c=c)
    #     fft_f = self.freq_symmetric(fft_f)  
    #     fft_data = fftpack.ifft(fft_f).real *freqend
    #     fft_times = np.fft.fftfreq(len(fft_data), d=freqstep)
    #     fft_data = fft_data[fft_times >= 0]
    #     fft_times = fft_times[fft_times >= 0]
    #     self.fft_times = fft_times
    #     self.fft_data = fft_data
    #     return fft_times, fft_data

    # def pelton_res_t_fft(self, res0=None, eta=None, tau=None,c=None,
    #     freq= None):
    #     res0 = self.get_param(res0, self.res0)
    #     eta = self.get_param(eta, self.eta)
    #     tau = self.get_param(tau, self.tau)
    #     c = self.get_param(c, self.c)   
    #     if freq is None:
    #         freq= self.generate_frequency(tau)
    #     self.freq = freq
    #     freqend = freq[-1]
    #     freqstep = freq[1] - freq[0]
    #     fft_f = self.pelton_res_f(freq=freq, res0=res0, eta=eta, tau=tau, c=c)
    #     fft_f = self.freq_symmetric(fft_f)  
    #     fft_data = fftpack.ifft(fft_f).real *freqend
    #     fft_times = np.fft.fftfreq(len(fft_data), d=freqstep)
    #     fft_data = fft_data[fft_times >= 0]
    #     fft_times = fft_times[fft_times >= 0]
    #     self.fft_times = fft_times
    #     self.fft_data = fft_data
    #     return fft_times, fft_data

    # def pelton_res_t(self, times, res0, eta,  tau, c, num_terms=100):
    #     """
    #     Parameters:
    #     - num_terms: Number of terms to approximate the infinite sum
    #     """
    #     term = np.zeros_like(times)
    #     sum_series = np.zeros_like(times)
    #     for n in range(num_terms):
    #         term = ((-1)**n /tau/ gamma(n * c + 1)) * (times / tau)**(n * c)
    #         sum_series += term
    #     rho_t = eta * res0 * sum_series
    #     if any(time == 0 for time in times):
    #         time_step = times[1] - times[0]
    #         ind_0 = (times == 0)
    #         res8 = res0 * (1.0 - eta)
    #         rho_t[ind_0] = res8 * 1.0 / time_step
    #     return rho_t
