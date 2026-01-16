"""This file contains some common utility functions used in ASC."""

import logging
import csv

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy import signal


def log_attributes(instance, additional_attributes: Optional[Dict[str, Any]] = None):
    """This function logs all relevant attributes of the given instance.

    Args:
      instance: The instance to log.additional_attributes
      additional_attributes: Additional attributes to log.
    """
    cls_name = type(instance).__name__
    # Also log small numpy arrays
    additional_attributes = additional_attributes or {}
    attributes = {**vars(instance), **additional_attributes}
    for name, attr in attributes.items():
        if (isinstance(attr, (int, float)) or
           (isinstance(attr, np.ndarray) and attr.nbytes < 32)):
            logging.debug('%s is using a %s of %d', cls_name, name, attr)


def faster_sosfilt(sos, x, zi):
    """Copy inputs and go directly to the cython implementation."""
    x_shape = x.shape
    zi_shape = zi.shape
    x_dtype = x.dtype
    x = np.array([x], order='C', dtype=np.float64)
    zi = np.array([zi], order='C')
    try:
        signal._sosfilt._sosfilt(sos, x, zi)  # modifies inputs in place
    except ValueError as e:
        raise ValueError(f'With {sos.shape=}, {x.shape=}, {zi.shape=}') from e
    x.shape = x_shape
    zi.shape = zi_shape
    return x.astype(x_dtype), zi


def get_delimiter(file_path: str) -> str:
    with open(file_path, 'r') as csvfile:
        delimiter = str(csv.Sniffer().sniff(csvfile.read()).delimiter)
        return delimiter


def noise_from_sqrt_psd(files, fs, dur, _rng_state):
    time_series = 0.
    for file in files:  # sum noises
        frequencies = np.linspace(0, fs // 2, dur * fs // 2 + 1)

        norm = 0.5 * dur ** 0.5

        # Fourier-amplitudes of white noise
        re = _rng_state.normal(0, norm, len(frequencies))
        im = _rng_state.normal(0, norm, len(frequencies))
        wtilde = re + 1j * im

        if isinstance(file, list):          # transfer function is specified
            n_file = file[0].strip()
            tf_file = file[1].strip()

            delimiter = get_delimiter(tf_file)
            tf_data = pd.read_csv(tf_file, names=['ff', 'tf'], delimiter=delimiter, skipinitialspace=True)
            ff = np.array(tf_data[['ff']].values.flatten())
            tf = np.array(list(map(complex, tf_data[['tf']].values.flatten())))
            tf = np.interp(frequencies, ff, tf, left=0, right=0)
        else:                               # no transfer function is specified
            tf = 1.
            n_file = file.strip()

        delimiter = get_delimiter(n_file)
        sqrt_psd_data = np.genfromtxt(n_file, delimiter=delimiter)

        rpsd = np.interp(frequencies, sqrt_psd_data[:, 0], sqrt_psd_data[:, 1], left=0, right=0)

        ctilde = wtilde * rpsd * tf

        # set DC = 0
        ctilde[0] = 0

        time_series += np.fft.irfft(ctilde) * fs

    return time_series


def noise_from_psd(files, fs, dur):
    time_series = 0.
    for file in files:  # sum noises
        delimiter = get_delimiter(file)
        psd_data = np.genfromtxt(files[k], delimiter=delimiter)

        frequencies = np.linspace(0, fs // 2, dur*fs // 2 + 1)

        norm = 0.5 * dur**0.5

        # Fourier amplitudes of white noise
        re = self._rng_state.normal(0, norm, len(frequencies))
        im = self._rng_state.normal(0, norm, len(frequencies))
        wtilde = re + 1j * im

        rpsd = np.interp(frequencies, psd_data[:, 0], np.sqrt(psd_data[:, 1]), left=0, right=0)
        ctilde = wtilde * rpsd

        # set DC = 0
        ctilde[0] = 0

        time_series += np.fft.irfft(ctilde) * fs

    return time_series


def compute_psd(
    timeseries: np.ndarray,
    t_fft: float,
    fs: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute the PSD of a signal.

    Args:
        timeseries: Time series of measurement values
        t_fft: Window length in seconds of the FF window.
        fs: Frequency sampling rate of the measurement.
    Returns:
        a tuple of:
        * Array of sample frequencies.
        * Power spectral density or power spectrum of x.
        * Root mean square of the PSD.
    """
    n_fft = int(t_fft * fs)
    if n_fft > timeseries.shape[0]:
        n_fft = timeseries.shape[0]
        t_fft = n_fft/fs
        logging.info(
            'Timeseries too short for chosen FFT window, adjusting to %d', t_fft)

    # note that beta>35 does not give you more sidelobe suppression
    window = signal.windows.kaiser(n_fft, beta=35)
    try:
        ff, psd = signal.welch(
            timeseries, fs=fs, window=window, nperseg=n_fft, noverlap=n_fft // 2)
    except ValueError as e:
        raise ValueError(
            f'Timeseries:{timeseries.shape}, {t_fft=}, {fs=}') from e

    rms = np.sqrt(1. / t_fft * np.sum(psd))

    return ff, psd, rms


def decimate(timeseries, fs, output_fs):
    n_down = fs//output_fs
    if n_down > 10:
        for k in range(len(timeseries)):
            timeseries[k][0] = signal.decimate(timeseries[k][0], 8, axis=0)
            timeseries[k][0] = signal.decimate(timeseries[k][0], n_down//8, axis=0)
    else:
        for k in range(len(timeseries)):
            timeseries[k][0] = signal.decimate(timeseries[k][0], n_down, axis=0)

    return timeseries


class FilterLP:

    def __init__(self, data, low_pass, size):
        """Use size=2 for Lightsaber and size=3 for Lightsaber_MC."""
        self.fs = data['simulation_sampling_frequency']

        f_pass = low_pass['pass-band_edge']
        f_stop = low_pass['stop-band_edge']
        min_att = low_pass['minimum_attenuation']

        ## low-pass
        [n, fn] = signal.ellipord(f_pass, f_stop, 1, min_att, fs=self.fs)
        print('Filter order:',n)

        zz, pp, k = signal.ellip(n, 1., min_att, 2*np.pi*fn, analog=True, output='zpk')
        zpk = signal.bilinear_zpk(zz, pp, k, self.fs)
        self.low_pass_sos = signal.zpk2sos(*zpk)
        self.low_pass_sos_state = np.zeros((size, 1, 2))
        self.low_pass_output = np.zeros(size,)

    def sample(self, input_signal=None):
        if input_signal is None:
          input_signal = np.zeros_like(self.low_pass_output)

        for i in range(self.low_pass_output.shape[0]):
            output, zf = faster_sosfilt(self.low_pass_sos, np.array([input_signal[i]]), zi=self.low_pass_sos_state[i])
            self.low_pass_sos_state[i] = zf

            self.low_pass_output[i] = output[0]

        return self.low_pass_output


class Postprocessing:
    """Strain noise filtering. Since the frequencies below 10 Hz and above 40 Hz would dominate during the
    reward process, and controls noise is relevant between 10 Hz and 25 Hz we need to whiten the strain noise
    in order that rewards are dominated by the noise in this frequency band."""

    def __init__(self, timeseries, fs, output_fs):

        self.timeseries = timeseries
        self.fs = fs
        self.output_fs = output_fs
        self.band_pass()
        self.band_pass_sos_state = np.zeros((12, 2))

    def band_pass(self):
        z1 = np.array([1 + 0j, 1 - 0j, 0.99 + 0j, 0.99 - 0j, 1.01 + 0j, 1.01 - 0j, 1 + 0j, 1 - 0j, 1 + 0j, 1 - 0j, 1 + 0j, 1 - 0j])
        p1 = np.array([-2*np.pi*5*1 + 2*np.pi*5*1j, -2*np.pi*5*1 - 2*np.pi*5*1j, -1.99*np.pi*5*1 + 1.99*np.pi*5*1j, -1.99*np.pi*5*1 - 1.99*np.pi*5*1j,
                    -2*np.pi*5*1 + 2*np.pi*5*1j, -2*np.pi*5*1 - 2*np.pi*5*1j, -2*np.pi*5*1 + 2*np.pi*5*1j, -2*np.pi*5*1 - 2*np.pi*5*1j, -2*np.pi*5*1 + 2*np.pi*5*1j, -2*np.pi*5*1 - 2*np.pi*5*1j,
                    -2*np.pi*40*1 + 2*np.pi*40*1j, -2*np.pi*40*1 - 2*np.pi*40*1j, -2*np.pi*40*1 + 2*np.pi*40*1j, -2*np.pi*40*1 - 2*np.pi*40*1j,
                    -2*np.pi*40*1 + 2*np.pi*40*1j, -2*np.pi*40*1 - 2*np.pi*40*1j, -1.99*np.pi*40*1 + 1.99*np.pi*40*1j, -1.99*np.pi*40*1 - 1.99*np.pi*40*1j,
                    -2*np.pi*40*1 + 2*np.pi*40*1j, -2*np.pi*40*1 - 2*np.pi*40*1j, -2*np.pi*40*1 + 2*np.pi*40*1j, -2*np.pi*40*1 - 2*np.pi*40*1j, -2*np.pi*40*1 + 2*np.pi*40*1j, -2*np.pi*40*1 - 2*np.pi*40*1j])
        k1 = 1.2e31

        zpk = signal.bilinear_zpk(z1, p1, k1, self.fs)
        self.band_pass_sos = signal.zpk2sos(*zpk)

    def sample(self, strain_noise=None):

        if strain_noise is not None:
            output, zf = faster_sosfilt(self.band_pass_sos, np.array([strain_noise]), zi=self.band_pass_sos_state)
            self.band_pass_sos_state = zf

        return output[0]
