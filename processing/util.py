import numpy as np
import torch

## -------------------------------------------------------------------------------------------------
##
## sine wave generators
##

def gen_sine(f_sampling, f_signal, num_periods=2, phase_rad=0):
    ''' generate num_periods of a sine wave, and take samples over it
        at f_sampling (every 1/f_sampling sec). The num of samples
        is f_sampling * Dt where Dt= num_periods/f_signal.
        @param f_sampling: sampling frequency
        @param f_signal: sinusoid frequency
        @param num_periods: num of periods of the sinusoid to generate samples for
        @param phase_rad: initial phase shift of the sinusoid
        @return:
        - signal samples
        - x indexes of the samples
        - time axis ticks corespondingg to the indexes of the samples
    '''

    # time points: T_signal = 1/f_signal, we want num_periods*T_signal with ticks (samples) every 1/f_sampling
    t = np.arange(0, num_periods * 1/f_signal - 1/f_sampling, 1/f_sampling)

    # compute the value (amplitude) of the sin wave at the for each sample
    w =2*np.pi*f_signal
    y = np.sin(w * t + phase_rad)

    return t,y

def gen_noisy_sine(*arglist):
    ''' same as gen_sine() but with added random noise
    '''
    t,y = gen_sine(*arglist)
    y = np.random.rand(len(y)) + y
    return t,y


def gen_sine_1sec(f_sampling, f_signal, phase_rad=0):
    ''' generate a 1s sine wave and take a fixed number of samples over a set num of
        periods (== f_signal) of the signal. Assuming a fixed 1s wnd of time for the
        sampled signal, then the num of sample points equals f_sampling and the num of
        periods of the sine signal equals f_signal
        @param f_sampling: sampling freq
        @param f_signal: sinusoid freq
        @param phase_rad: initial phase shift
        @return:
        - signal samples
        - x indexes of the samples
        - time axis ticks corespondingg to the indexes of the samples
    '''
    #f_sampling = 100 # sample rate
    #f_signal = 2 # the frequency of the signal

    # num samples we want generated at f_sampling Hz
    x = np.arange(f_sampling) # num sample points in 1s
    t = x/f_sampling # time points (s) corresponding to the sample points

    # compute the value (amplitude) of the sin wave at the each sample point
    w = 2*np.pi*f_signal # polar angle
    #y = np.sin( w * x/f_sampling + phase_rad)
    y = np.sin( w * t + phase_rad)

    return x,t,y


## -------------------------------------------------------------------------------------------------
##
## convert to DB scale (useful for the fft magnitude
##
def to_db(vals):
    '''
    First get rid of the zero values (replace them with something infinitesimaly small,
    then convert to db scale
    @param vals
    @return: vals in db
    '''
    epsilon = 0.00000000001
    vals_stable = np.where(vals == 0, epsilon, vals)
    return 10*np.log10(vals_stable)


def to_db_torch(vals):
    '''
    First get rid of the zero values (replace them with something infinitesimaly small,
    then convert to db scale
    @param vals
    @return: vals in db
    '''
    epsilon = 0.00000000001
    vals_stable = torch.where(vals == 0, epsilon, vals)
    return 10*torch.log10(vals_stable)
