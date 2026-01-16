"""This module contains common plotting functionalities."""

import os
from typing import Optional

import utils

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal


def plot_psd(timeseries: np.ndarray,
             duration_fft: int,
             sampling_frequency: int,
             filename: str,
             ylabel: str = 'Spectrum [Hz$^{-1/2}$]'):

    plt.figure()
    if timeseries.ndim > 1:
        for k in range(timeseries.shape[1]):
            ff, psd, rms = utils.compute_psd(timeseries[:, k], duration_fft, sampling_frequency)
            fi1 = np.argmin(np.abs(ff - 0.1))
            fi2 = np.argmin(np.abs(ff - 100))
            plt.loglog(ff[fi1:fi2], np.sqrt(psd[fi1:fi2]), label='rms = {:5.2e}'.format(rms))
    else:
        ff, psd, rms = utils.compute_psd(timeseries, duration_fft, sampling_frequency)
        fi1 = np.argmin(np.abs(ff - 0.1))
        fi2 = np.argmin(np.abs(ff - 100))
        plt.loglog(ff[fi1:fi2], np.sqrt(psd[fi1:fi2]), label='rms = {:5.2e}'.format(rms))
    plt.xlim(0.1, 100)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, which='both')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_hoft(h_noise,
              duration_fft,
              fs,
              label,
              filename,
              reference_data_file=None,
              ylimit=None):
  if reference_data_file is not None:
    dn = pd.read_csv(
        reference_data_file,
        names=['ff', 'susT', 'coatT', 'quantum', 'aplus'],
        delimiter=' ',
        skipinitialspace=True)
    ff = np.array(dn[['ff']].values.flatten())
    aplus = np.array(dn[['aplus']].values.flatten())

  plt.figure()
  for i in range(len(h_noise[0, :])):
    ff_data, psd, _ = utils.compute_psd(h_noise[:, i], t_fft=duration_fft, fs=fs)
    plt.loglog(ff_data, np.sqrt(psd), label=label[i])

  if reference_data_file is not None:
    plt.loglog(ff, aplus, label='AdV LIGO +')
  plt.xlim(1, 100)
  if ylimit is not None:
    plt.ylim(*ylimit)  # arm cavity has 1e-25, 3e-22
  plt.xlabel('Frequency [Hz]')
  plt.ylabel('Strain noise [Hz$^{-1/2}$]')
  plt.legend()
  plt.grid(True, which='both')
  plt.tight_layout()
  if filename is not None:
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_diff_disp_noise(deltaL,
                         T_fft,
                         fs,
                         label,
                         filename: Optional[str] = None):

  plt.figure()
  for i in range(len(deltaL[0, :])):
    ff_data, psd, _ = utils.compute_psd(deltaL[:, i], T_fft, fs)
    plt.loglog(ff_data, np.sqrt(psd), label=label[i])

  plt.xlim(10, 100)
  plt.ylim(1e-21, 4e-17)
  plt.xlabel('Frequency [Hz]')
  plt.ylabel('Differential displacement noise [m/Hz$^{1/2}$]')
  plt.legend()
  plt.grid(True, which='both')
  plt.tight_layout()
  if filename is not None:
    plt.savefig(filename, dpi=300)
    plt.close()


def sos_freq_resp(sos_sys: np.ndarray,
                  fs: int,
                  filename: Optional[str]):
  f, h = signal.sosfreqz(sos_sys, worN=100000, fs=fs)
  fi1 = np.argmin(np.abs(f-0.1))
  fi2 = np.argmin(np.abs(f-100))
  plt.figure()
  plt.subplot(2, 1, 1)
  plt.semilogx(
      f[fi1:fi2],
      20 * np.log10(np.abs(h[fi1:fi2])),
      alpha=0.8)
  plt.xlabel('Frequency [Hz]')
  plt.ylabel('tf, mag [dB]')
  plt.xlim(0.1, 100)
  plt.grid(True, which='both')
  plt.subplot(2, 1, 2)
  plt.semilogx(
      f[fi1:fi2],
      np.unwrap(np.angle(h[fi1:fi2], deg=True), discont=179),
      alpha=0.8)
  plt.xlabel('Frequency [Hz]')
  plt.ylabel('tf, phase [deg]')
  plt.xlim(0.1, 100)
  plt.grid(True, which='both')
  plt.tight_layout()
  if filename is not None:
    plt.savefig(filename, dpi=300)
    plt.close()


def transfer_function(sos_sys: np.ndarray,
                      T: int,
                      fs: int,
                      T_fft: int = 64,
                      ylabel: str = 'Transfer function',
                      filename: Optional[str] = None):

  # Fourier amplitudes of white noise (not the best choice!!)
  re = np.random.normal(0, 1, T * fs // 2 + 1)
  im = np.random.normal(0, 1, T * fs // 2 + 1)
  wtilde = re + 1j * im
  wtilde[0] = 0

  input_signal = np.fft.irfft(wtilde) * fs

  tt = np.linspace(0, T, len(input_signal) + 1)
  tt = tt[0:-1]

  state = signal.sosfilt_zi(sos_sys)
  output, _ = signal.sosfilt(sos_sys, input_signal, zi=state)

  n_fft = T_fft * fs
  window = signal.windows.hann(
      n_fft)  # note that beta>35 does not give you more sidelobe suppression
  ff, pxy = signal.csd(
      input_signal,
      output,
      fs=fs,
      window=window,
      nperseg=n_fft,
      noverlap=n_fft // 2)
  ff, pxx = signal.welch(
      input_signal, fs=fs, window=window, nperseg=n_fft, noverlap=n_fft // 2)

  tf = pxy / pxx

  fi = np.logical_and(
      ff > 0.1, ff < 100
  )  # constrain plotted values since this leads to better automatic y-range in the plot
  plt.figure()
  plt.subplot(2, 1, 1)
  plt.semilogx(ff[fi], 20 * np.log10(np.abs(tf[fi])))  # Bode magnitude plot
  plt.xlabel('Frequency [Hz]')
  plt.ylabel(ylabel + ', mag [dB]')
  plt.xlim(0.1, 100)
  plt.grid(True, which='both')
  plt.subplot(2, 1, 2)
  plt.semilogx(ff[fi], np.unwrap(np.angle(tf[fi]) * 180. / np.pi,
                                 discont=179))  # Bode phase plot
  plt.xlabel('Frequency [Hz]')
  plt.ylabel(ylabel + ', phase [deg]')
  plt.xlim(0.1, 100)
  plt.grid(True, which='both')
  plt.tight_layout()
  if filename is not None:
    plt.savefig(filename, dpi=300)
    plt.close()
