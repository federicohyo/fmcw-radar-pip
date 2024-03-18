import numpy as np
import scipy.fft as fft

## -------------------------------------------------------------------------------------------------
##
## 1D FFT all-tasks-inclusive function
##

def run_fft(s, n, hann_taper= True, phase_in_deg=True, f_sampling=None, zero_dc=False, shift_fft=True):
    '''
    Apply 1D n point FFT along the x-axis to a signal or array of signals s.
    @param hann_taper: if == True, before the FFT it applies a Hann window to s
    @param phase_in_deg: if == True, phase is computed in degrees, else in rad
    @param f_sampling: if provided the sampling freq, reports freq resolution and Nyquist-Shannon freq, and x-axis is computed in Hz
    @param zero_dc: if == True the DC component is removed (increases the resolution of the remaining components)
    @returns
    - the complex FFT values
    - the magnitude
    - the phase
    - the x-axis normalised in [0,1] x Nyquist
    - the frequencies
    '''

    if s.ndim == 1:
        s = s.reshape([-1, s.shape[0]])

    rad2degrees = lambda rad: rad * 180/np.pi    # for conversion of rad to deg in phase


    ## --- remove DC

    ## basic step that often seems to work miracles in revealing the spectrum is to substract
    ## the mean from the time domain signal (equiv of zeroing the DC component in freq dom)
    if zero_dc is True:
        s = s - np.mean(s, axis=1).reshape([-1,1])


    ## --- taper to reduce spectral leakage in spectrum (sharp edge effects when signal end-to-start has a discontinuity -- FFT assumes periodic signal)

    ## the canonical thing to do this is to deform the time domain signal with a window (taper) before
    ## FFT so that the "jump" between the signal edges is zero -Hann wnd- or very small -Hamming wnd-
    ## (or equiv to apply circular convolution to the spectrum with the FFT kernel of the resp wnd)
    ##
    ## NOTE: in the case of a clean modulo period sampled signal this will contaminate the spectrum
    ## with nz side components at each actual frequency (i.e. it will cause some leakage)
    ##
    if hann_taper is True:
        hann_wnd = np.hanning(s.shape[1])
        s = s * hann_wnd


    ## --- apply FFT

    ## FFT bins (freq points/resolution between DC and f_sampling/2). Increasing the bins beyond
    ## the time points of the sigmal, causes the time domain signal to be zero padded and increases
    ## the freq resolution (between DC and f_sampling/2), but not the precision of the transform
    ## (instead we have an artifical smoothing interpolation effect on the plot -- that does not
    ## represent however actual frequency content in these frequencies)
    ## NOTE: To avoid tears with numerical precision better use a power of 2. To avoid symmetric
    ## phase artifacts make equal or multiple of number of samples in signal. The latter might
    ## also reduce spectral leakage originating from misalignment of samples to freq bins.
    ##
    fft_N = n
    #print("num FFT bins:", fft_N)

    ## not shifted ver
    ## [0, f_sampling/2) in bins [0, fft_N/2)
    ## [-f_sampling/2, 0) in bins [fft_N/2, fft_N-1) -- these are the neg frequency components
    ##
    # x = np.arange(0, fft_N, 1) / fft_N           # normalize x-axis
    # s_fft = fft.fft(s, n=fft_N, axis=1)
    ##
    ## .. or .
    ##
    ## centered ver (double sided)
    ## [0, f_sampling/2) in bins [0, fft_N/2)
    ## [-f_sampling/2, 0) in bins [-fft_N/2, 0) -- neg frequency components
    ##
    x = np.arange(-fft_N/2, fft_N/2, 1) / fft_N     # normalize x-axis
    
    if shift_fft is True: 
        s_fft = fft.fftshift( fft.fft(s, n=fft_N, axis=1), axes=1 )
    else:
        s_fft = fft.fft(s, n=fft_N, axis=1)

    ## compute (normalized) magnitude
    ##
    s_fft_mag = np.abs(s_fft)                       # equal to  np.sqrt( np.real(s_fft)**2 + np.imag(s_fft)**2 )
    if hann_taper is not True:
        s_fft_mag = 1/s.shape[1] * s_fft_mag        # normalize magnitude by the num of sample points (NOT fft_N)
    else:
        s_fft_mag = 1/np.sum(hann_wnd) * s_fft_mag  # normalize magnitude by the hann wnd weights (power loss compensation)

    ## compute phase (after filtering out numerical rounding noise, that can mess up phase plot)
    ##
    s_fft_tmp = s_fft.copy()
    thres = 0.0001 * np.max(np.abs(s_fft_tmp))       # auto threshold to zero out very small vals (a dominant DC will have a serious effect here setting the thres very high)
    #thres = 0.0001                                  # manual threshold to zero out very small vals from numerical rounding errors
    s_fft_tmp = np.where(np.abs(s_fft_tmp) < thres, 0.0+0.0j, s_fft_tmp)              # zero both real/imag based on magnitude
    #s_fft_tmp.real = np.where(np.abs(s_fft_tmp.real) < thres, 0.0, s_fft_tmp.real)   # ... or zero separatelly the re part if it is too small
    #s_fft_tmp.imag = np.where(np.abs(s_fft_tmp.imag) < thres, 0.0, s_fft_tmp.imag)   #     and separatelly the im part if it is too small
    s_fft_phase = np.angle(s_fft_tmp)                # equal to  np.arctan2( np.imag(s_fft), np.real(s_fft) )


    ## --- conversions of units

    if phase_in_deg is True:
        s_fft_phase = rad2degrees( s_fft_phase )  # phase in degrees

    if f_sampling is not None:
        x = x * f_sampling       # x axis in Hz
        print("FFT frequency resolution:", f_sampling / fft_N, " Hz")
        print("Max FFT freq (Nyquist-Shannon):", f_sampling/2, " Hz")

    return s_fft, s_fft_mag, s_fft_phase, x


## -------------------------------------------------------------------------------------------------
##
## 2D FFT 'all-in-1' function
##

def run_fft_2d(s, n_x, n_y, hann_taper=True, phase_in_deg=False, zero_dc=False, correct_mag_phase=False, shift_fft=True):
    '''
    Apply 2D n_x by n_y point FFT first along the x-axis and then the y-axis on a 2D signal array s.
    The array is supposed to be hold a wrapped signal along the rows. I.e. fast time on x-dim and
    slow time on y_dim.
    @param hann_taper: if == True, it applies a Hann window to s row-wise before the 1st FFT
    @param phase_in_deg: if == True, phase is computed in degrees, else in rad
    @param zero_dc: if  == True the DC component is removed before 1st FFT row-wise (increases the resolution of the remaining components)
    @param correct_mag_phase: if == True the correctly scaled versions of magnitude and phase are returned, else not correction is applied
    @returns
    - the complex 2D FFT values (shifted low freq in the centre)
    - the magnitude
    - the phase
    - the x-axis normalised in [0,1] x Nyquist
    - the y-axis normalised in [0,1] x Nyquist
    '''

    if s.ndim == 1:
        raise ValueError("param s must be 2d array")

    fft1_s , _, _,                       x = run_fft(s, n_x, hann_taper=hann_taper, phase_in_deg = False, zero_dc=zero_dc, shift_fft=shift_fft)
    fft2_s_, fft2_s_mag_, fft2_s_phase_, y = run_fft(fft1_s.T, n_y, hann_taper=hann_taper, phase_in_deg = phase_in_deg, zero_dc=False, shift_fft=shift_fft)
    fft2_s = fft2_s_.T

    if correct_mag_phase is True:       # we ll return the correctly scaled version (this has been taken care of inside run_fft() )
        print('correcting magnitude and phase')
        fft2_s_mag = fft2_s_mag_.T
        fft2_s_phase = fft2_s_phase_.T
    else:
        print('not correcting magnitude and phase')
        fft2_s_mag = np.abs(fft2_s)
        fft2_s_phase = np.angle(fft2_s)

    return fft2_s, fft2_s_mag, fft2_s_phase, x, y



## -------------------------------------------------------------------------------------------------
##
## Function to run STFT and generate spectrograms
##


def run_stft(s, n, wnd_len = 10, wnd_stride = 1, hann_taper = True, bin_idx = None, axis = 1):
    '''
    Apply n point STFT to along each of the cols (axis=0) or rows (axis=1) listed in bin_idx of s and
    return the resp list of spectrograms.
    @param hann_taper: specifies whether to apply a Hanning window before the FFT
    @param wnd_len: is the size of the sliding window
    @param wnd_stride: defines the window overlap (num of positions window is moved at each time step)
    @returns
    - a list of stft transforms, one foreach bin
    - a list of spectrograms, one foreach bin
    '''

    # Given arguments of the function run_stft
    print("s.shape:", s.shape)
    print("n:", n)
    print("wnd_len:", wnd_len)
    print("wnd_stride:", wnd_stride)
    print("hann_taper:", hann_taper)
    print("bin_idx length:", len(bin_idx))
    print("axis:", axis)

    if s.ndim == 1:
        s = s.reshape([1, -1])  # convert to 2D


    if axis == 0:
        s = s.T     # we ll parse always along rows
        axis = 1


    if bin_idx is None:
        bin_idx = list(range(s.shape[0]))
    elif np.max(bin_idx) > s.shape[0]-1:
        raise ValueError("Range-bin index " + str(np.max(bin_idx)) + " out of range");


    if wnd_stride == 0:
        raise ValueError("Window stride but must not be 0 otherwise STFT is pointless");

    last_data_idx = s.shape[1]-1


    if wnd_len + wnd_stride >= s.shape[1]:
        zero_pad = np.zeros( [s.shape[0], wnd_len + wnd_stride - s.shape[1]] )
        print ("Zero padding", zero_pad.shape[1], "will be added to ensure at least 2 time points in spectrogram");
        s = np.concatenate( (s, zero_pad), axis=1)

    print("s.shape after pad:", s.shape)

    fft_N = n
    stft_list =  []
    spectrogram_list = []


    for b in bin_idx:    # TODO vectorize this outer loop. Since run_fft() is vectorized it can process all bins in parallel
        print("bin:", b)

        wnd_start =0

        stft_mat = np.zeros([fft_N ,1])
        spectrogram = np.zeros([fft_N ,1])

        while wnd_start <= last_data_idx:

            wnd_end = wnd_start + wnd_len
            # print("wnd_start:", wnd_start)
            # print("wnd_end:", wnd_end)

            stft_slice, stft_mag_slice, _, _ = run_fft(s[b, wnd_start:wnd_end], fft_N, hann_taper=hann_taper, phase_in_deg = False, zero_dc=False)
            # print("stft_slice.shape:", stft_slice.shape)
            # print("stft_mag_slice.shape:", stft_mag_slice.shape)

            spectrogram = np.concatenate( (spectrogram, stft_mag_slice.T), axis=1)
            stft_mat = np.concatenate( (stft_mat, stft_slice.T), axis=1)

            wnd_start += wnd_stride

        spectrogram_list.append( spectrogram[:, 1:] )
        stft_list.append( stft_mat[:, 1:] )

        print("spectogram lenght:", len(spectrogram_list), spectrogram_list[0].shape)
        print("stft lenght:", len(stft_list), stft_list[0].shape)

    return stft_list, spectrogram_list


# ---------------------------

fft_N_range = 512    # num range bins
fft_N_mdopp = 256

range_bins = list(range(0,50))

remove_dc = False    # when True the pattern in range-doppler magnitude is more pronounced but there is no pattern in the phase. When False there is a pattern in the range-doppler phase
apply_window = True  # if False the doppler pattern disappears and the log(magnitude) in range gives many divisions-by-zero

wnd=fft_N_mdopp
stride=wnd//2     # when applying a Hann window b4 STFT, a stride <= wnd//2 has the nice effect that only the beginning and the end of the signal are attenuated but nowhere in-between
# stride=3*wnd//4

def range_mag(input, fft_Nx):
    encoded_frames = []

    for frame in input:
        _, range_fft_mag, _, _ = run_fft (frame, fft_Nx, hann_taper = apply_window, zero_dc=remove_dc)
        encoded_frames.append(range_fft_mag)

    return encoded_frames

def range_phase(input, fft_Nx):
    encoded_frames = []

    for frame in input:
        _, _, range_fft_phase, _ = run_fft (frame, fft_Nx, hann_taper = apply_window, zero_dc=remove_dc)
        encoded_frames.append(range_fft_phase)

    return encoded_frames

def doppler_mag(input, fft_Nx, fft_Ny):
    encoded_frames = []

    for frame in input:
        _, doppler_fft_mag, _, _, _ = run_fft_2d(frame, fft_Nx, fft_Ny, hann_taper=apply_window, zero_dc=remove_dc, correct_mag_phase=True)
        encoded_frames.append(doppler_fft_mag)

    return encoded_frames

def doppler_phase(input, fft_Nx, fft_Ny):
    encoded_frames = []

    for frame in input:
        _, _, doppler_fft_phase, _, _ = run_fft_2d(frame, fft_Nx, fft_Ny, hann_taper=apply_window, zero_dc=remove_dc, correct_mag_phase=True)
        encoded_frames.append(doppler_fft_phase)

    return encoded_frames

def mu_doppler(input):

    # input shape
    print("input shape:", input.shape)
    print("fft_N_range:", fft_N_range)
    print("fft_N_mdopp:", fft_N_mdopp)

    all_frames_fft = np.zeros([1, fft_N_range])
    print('all_frames_fft shape:', all_frames_fft.shape)

    for frame in input:
        frame_fft, _, _, _ = run_fft(frame, fft_N_range, hann_taper = True, zero_dc=False)
        all_frames_fft = np.concatenate([all_frames_fft, frame_fft], axis = 0)

    all_frames_fft = all_frames_fft[1:, :]    # get rid of the first/empty row

    ## micro-doppler STFT
    # creating a list from 256 until 305 with a step of 1
    mdopp_bins = [fft_N_range//2 + range_bin for range_bin in range_bins]
    print('mdopp_bins' , len(mdopp_bins))
    print(mdopp_bins)
    _, spgrams = run_stft(all_frames_fft, n = fft_N_mdopp, wnd_len = wnd, wnd_stride = stride, hann_taper=True, bin_idx=mdopp_bins, axis = 0)

    print()

    # select correct muDoppler range bin - naive way -> bin with highest mean...
    # we could also select the right bin by knowing the distance, but that is information we might not have during deployment...
    
#    highest_mean_bin = 0
#    highest_mean = 0
#
#    for index, spgram in enumerate(spgrams):
#        mean = np.mean(spgram)
#        if(mean > highest_mean): 
#            highest_mean_bin = index
#            highest_mean = mean
#
#    return spgrams[highest_mean_bin]

    # discard first frames (they only have DC? why?)
    # return spgrams[2:]
    return spgrams