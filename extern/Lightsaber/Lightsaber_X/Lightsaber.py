"""Lightsaber is an ASC time-domain simulator to test novel feedback-filter designs.

Produced by Jan Harms and Tomislav Andric

Collaborators Rana Adhikari and Hang Yu from Caltech provided all the insight and data for the ASC modeling.

version 11.0 (18-March-2022) time dependent plant with Sidles-Sigg compensation.
Lightsaber implements pitch dynamics with noise inputs from ISI-L and TOP NL/NP from damping OSEMs. The dynamics
include a power-dependent Sidles-Sigg torque feedback. Lightsaber simulates the test-mass pitch soft-hard mode readout.
In lack of a state-space/SOS model for the ISI/TOP input noises, they are produced by Fourier methods in fixed-size
batches.
"""

import os
import sys

import dataclasses
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as signal
import tqdm

import utils
import plotting


class System:
    def __init__(self):
        self.components = []
        self.component_types = []

    def append(self, component):
        self.components.append(component)
        self.component_types.append(component.type)

    def get_by_type(self, component_type):
        selected = []
        for comp in self.components:
            if comp.type == component_type:
                selected.append(comp)
        return selected

    def get_by_name(self, component_name):
        selected = []
        for comp in self.components:
            if comp.name == component_name:
                selected.append(comp)
        return selected

    def reset_counters(self):
        for c in self.components:
            c.reset_counters()

    def step(self, inputs):
        outputs = inputs.copy()
        for comp in self.components:
            if isinstance(comp, Laser):
                comp.step()
                outputs['in_power'] = comp.out
            if isinstance(comp, Beam):
                comp.step(inputs['pitch'], inputs['in_power'])
                outputs['rad_torque'] = comp.out
            if isinstance(comp, Controller):
                comp.step(inputs['readout'])
                if comp.act_point == 'sus':
                    outputs['act_sus'] = comp.out
                else:
                    outputs['act_mirror'] = comp.out
            if isinstance(comp, Mirror):
                mirrors = self.get_by_type('Mirror')
                i = mirrors.index(comp)
                comp.step(inputs['rad_torque'][i], inputs['act_sus'][i], inputs['act_mirror'][i])
                outputs['pitch'][i] = comp.out
            if isinstance(comp, Sensor):
                comp.step(inputs['pitch'])
                outputs['readout'] = comp.out

        return outputs


class Laser:
    def __init__(self, config_data, seed=None, plot_dir=False):
        self._rng_state = np.random.RandomState(seed=seed)
        utils.log_attributes(self, dict(seed=seed))

        self.name = config_data['name']                                     # name of laser
        self.type = config_data['type']                                     # type of component (Sensor)
        self.P_av = eval(str(config_data['power']))                         # average laser power (unidirectional) [W]
        self.wavelength = eval(str(config_data['wave_length']))             # laser wavelength [m]
        self.fs = eval(str(config_data['simulation_sampling_frequency']))   # sampling frequency [Hz]
        self.dur_batch = eval(str(config_data['duration_batch']))           # duration of simulated batch [s]
        self.dur_fft = eval(str(config_data['duration_fft']))               # duration of FFT segment [s]

        # time series of laser power
        self.power_tt = (1+utils.noise_from_sqrt_psd([config_data['RIN']], self.fs, self.dur_batch, self._rng_state)) * self.P_av
        if plot_dir:
            plotting.plot_psd(self.power_tt / self.P_av, self.dur_fft, self.fs,
                              os.path.join(plot_dir, 'RIN_spectrum.png'),
                              ylabel='Relative input power fluctuations [Hz$^{-1/2}$]')

        self.P = 0.                 # instantaneous power (unidirectional) [W]
        self.out = None             # output value of this components
        self.input_ch = []          # names of input channels
        self.output_ch = {'power': 0}   # names of output channels

        self.ti = 0  # index running through input noise batch

    def setIO(self, ii_in, ii_out):
        self.output_ch['power'] = ii_out

    def step(self):
        self.P = self.power_tt[self.ti]
        self.ti += 1

        self.out = self.P

    def reset_counters(self):
        self.ti = 0

    def substitute_names_by_variables(self, comp):
        pass   # do nothing


class Beam:
    def __init__(self, config_data, seed=None, plot_dir=False):
        self._rng_state = np.random.RandomState(seed=seed)
        utils.log_attributes(self, dict(seed=seed))

        self.angle_to_bs = []                               # matrix mapping angular to beam-spot motion
        self.local_to_eigen = []                            # matrix mapping angular motion from local to eigenmode
        self.eigen_to_local = []                            # matrix mapping angular motion from eigenmode to local
        self.dydth_soft = []                                # pitch to beamspot, soft mode
        self.dydth_hard = []                                # pitch to beamspot, hard mode
        self.high_pass_sos = []                             # high-pass filter to mimic length control
        self.high_pass_sos_state = []                       # initial conditions cascaded filter delays
        self.wavelength = []                                # wave length of the laser [m]

        self.m1 = []                                                # first mirror connected to beam
        self.m2 = []                                                # second mirror connected to beam

        self.name = config_data['name']                                     # name of beam
        self.type = config_data['type']                                     # type of component (Sensor)
        self.BS_offset = np.array(eval(str(config_data['BS_offset'])))      # intentional beam-spot offset [m]
        self.L = eval(str(config_data['length']))                           # length of the beam [m]
        self.fs = eval(str(config_data['simulation_sampling_frequency']))   # sampling frequency [Hz]

        self.BS = np.array([0., 0.])    # beam spots [m]
        self.P = 0.                     # instantaneous power (bidirectional) [W]
        self.P_av = 0.                  # running average of beam power [W]
        self.dL = 0.                    # instantaneous arm-length change [m]
        self.ti = 0                     # index running through input noise batch
        self.N = 1.                     # norm for running average of cavity power
        self.out = None                 # output value of this components

        self.high_pass()

        self.input_ch = {'pitch': 0, 'power': 0}    # names of input channels
        self.output_ch = {'torque': 0}              # names of output channels

    def setIO(self, ii_in, ii_out):
        self.input_ch['pitch'] = ii_in[0]
        self.input_ch['power'] = ii_in[1]
        self.output_ch['tau'] = ii_out

    def set_parameters(self, m1, m2, wavelength):
        self.m1 = m1                                                # first mirror
        self.m2 = m2                                                # second mirror
        self.wavelength = wavelength

        g1 = 1 - self.L / self.m1.RoC
        g2 = 1 - self.L / self.m2.RoC
        r = 0.5 * (g1 - g2 + np.sqrt((g1-g2)**2 + 4))

        # matrix that connects beam spots with local angles
        self.angle_to_bs = self.L / (1-g1*g2) * np.array([[g2, 1], [1, g1]])

        # matrix that converts from local to eigenbasis of angular motion
        self.local_to_eigen = np.array([[1, r], [-r, 1]]) / (1 + r**2)

        # matrix that converts from eigenbasis to local angular motion
        self.eigen_to_local = np.array([[1, -r], [r, 1]])

        # conversion from soft/hard angular motion to beam-spot motion on mirrors
        # [note that I introduced a change of sign assuming that these components come from eigen_to_local*angle_to_bs]
        self.dydth_soft = (self.L/2) * ((g1+g2) + np.sqrt((g2-g1)**2 + 4)) / (1-g1*g2)
        self.dydth_hard = (self.L/2) * ((g1+g2) - np.sqrt((g2-g1)**2 + 4)) / (1-g1*g2)

    def high_pass(self):
        """
        This function is used to filter out the low-frequency content of the power fluctuations,
        thus simulating the effect of the DARM length control.
        """
        z, p, k = signal.ellip(2, 1., 140., 2.*np.pi*50., btype='high', analog=True, output='zpk')
        #k *= 10.**(1./20.) # why should this factor be necessary?

        zpk = signal.bilinear_zpk(z, p, k, self.fs)
        self.high_pass_sos = signal.zpk2sos(*zpk)
        self.high_pass_sos_state = np.zeros((len(self.high_pass_sos[:, 0]), 2))

    def step(self, pitch, input_power):
        self.BS = self.angle_to_bs @ pitch + self.BS_offset     # beam spots
        self.dL = np.sum(self.BS*pitch, axis=-1)                # length noise

        # applying high-pass filter to mimic length control
        dL_hp, zf = utils.faster_sosfilt(self.high_pass_sos, np.array([self.dL]), zi=self.high_pass_sos_state)
        self.high_pass_sos_state = zf

        # power in the cavity
        self.P = input_power * self.m1.T / np.abs(
            1 - np.sqrt(self.m1.R) * np.sqrt(self.m2.R) * np.exp(4j * np.pi * dL_hp[0] / self.wavelength)) ** 2

        # running average of cavity power
        self.P_av = self.P_av + 1 / self.N * (self.P - self.P_av)
        if self.N < 1000:
            self.N += 1

        # radiation-pressure torques
        torque_dc = 2 / 299792458.0 * self.P_av * self.BS_offset
        self.out = 2 / 299792458.0 * self.P * (self.BS+self.BS_offset) - torque_dc

        self.ti += 1

    def reset_counters(self):
        self.ti = 0

    def substitute_names_by_variables(self, comp):
        pass   # do nothing


class Mirror:
    def __init__(self, config_data, seed=None, plot_dir=False):
        self._rng_state = np.random.RandomState(seed=seed)
        utils.log_attributes(self, dict(seed=seed))

        self.name = config_data['name']                                     # name of mirror
        self.type = config_data['type']                                     # type of component (Sensor)
        self.RoC = eval(str(config_data['RoC']))                            # mirror radius of curvature [m]
        self.T = eval(str(config_data['T']))                                # power transmissivity
        self.fs = eval(str(config_data['simulation_sampling_frequency']))   # sampling frequency [Hz]
        self.T_batch = eval(str(config_data['duration_batch']))             # duration of simulated batch [s]
        self.T_fft = eval(str(config_data['duration_fft']))                 # duration of simulated batch [s]

        # response functions as second-order sections
        self.rad_to_angle_sos = []
        self.act_to_angle_sos = []

        self.rad_to_angle_sos_state = []                                    # initial conditions cascaded filter delays
        self.act_to_angle_sos_state = []                                    # initial conditions cascaded filter delays

        # time series of noise from suspensions (e.g., platform + damping loop)
        self.sus_noise_tt = utils.noise_from_sqrt_psd(config_data['sus_noise'], self.fs, self.T_batch, self._rng_state)

        self.ti = 0             # index running through input noise batch
        self.P = 0              # current pitch angle [rad]
        self.out = None         # output value of this components
        self.R = 1 - self.T     # power reflectivity

        self.set_sos_models(config_data, plot_dir)

        if plot_dir:
            plotting.plot_psd(self.sus_noise_tt, self.T_fft, self.fs, os.path.join(plot_dir, 'sus_spectrum.png'),
                              ylabel='Suspension noise, TM P [rad/Hz$^{1/2}$]')

    def set_sos_models(self, config_data, plot_dir):
        zz = np.array([eval(str(comp)) for comp in config_data['act_to_angle'][0]])
        pp = np.array([eval(str(comp)) for comp in config_data['act_to_angle'][1]])
        k = eval(str(config_data['act_to_angle'][2][0]))

        zpk = signal.bilinear_zpk(zz, pp, k, self.fs)
        self.act_to_angle_sos = signal.zpk2sos(*zpk)
        self.act_to_angle_sos_state = np.zeros((len(self.act_to_angle_sos[:, 0]), 2))

        zz = np.array([eval(str(comp)) for comp in config_data['rad_to_angle'][0]])
        pp = np.array([eval(str(comp)) for comp in config_data['rad_to_angle'][1]])
        k = eval(str(config_data['rad_to_angle'][2][0]))

        zpk = signal.bilinear_zpk(zz, pp, k, self.fs)
        self.rad_to_angle_sos = signal.zpk2sos(*zpk)
        self.rad_to_angle_sos_state = np.zeros((len(self.rad_to_angle_sos[:, 0]), 2))

        if plot_dir:
            plotting.sos_freq_resp(self.act_to_angle_sos, self.fs, os.path.join(plot_dir, f'bode_act_to_angle.png'))
            plotting.sos_freq_resp(self.rad_to_angle_sos, self.fs, os.path.join(plot_dir, f'bode_rad_to_angle.png'))

    def step(self, rad_torque=0., act_upper_stage=0., act_mirror=0.):

        # angular motion from radiation-pressure torque acting on mirror
        angle_torque_mirror, zf = utils.faster_sosfilt(self.rad_to_angle_sos, np.array([rad_torque+act_mirror]),
                                                       zi=self.rad_to_angle_sos_state)
        self.rad_to_angle_sos_state = zf

        # angular motion from angular controls acting on upper suspension stage
        angle_torque_upper, zf = utils.faster_sosfilt(self.act_to_angle_sos, np.array([act_upper_stage]),
                                                      zi=self.act_to_angle_sos_state)
        self.act_to_angle_sos_state = zf

        # add inputs to angular motion (contains radiation-pressure compensation)
        self.P = self.sus_noise_tt[self.ti] + angle_torque_mirror[0] + angle_torque_upper[0]
        self.out = self.P

        self.ti += 1

    def reset_counters(self):
        self.ti = 0

    def substitute_names_by_variables(self, comp):
        self.ti = self.ti   # do nothing


class Sensor:
    def __init__(self, config_data, seed=None, plot_dir=False):
        self._rng_state = np.random.RandomState(seed=seed)
        utils.log_attributes(self, dict(seed=seed))

        self.name = config_data['name']                                     # name of sensor
        self.type = config_data['type']                                     # type of component (Sensor)
        self.fs = eval(str(config_data['simulation_sampling_frequency']))   # sampling frequency [Hz]
        self.T_batch = eval(str(config_data['duration_batch']))             # duration of simulated batch [s]
        self.T_fft = eval(str(config_data['duration_fft']))                 # duration of FFT segment [s]
        self.matrix = config_data['matrix']                                 # matrix mapping input channels to signals

        # produce time series of sensor noise
        self.readout_noise_tt = []
        noise_files = config_data['readout_noise'].split(',')
        self.readout_noise_tt = np.vstack((
            utils.noise_from_sqrt_psd([noise_files[0].strip()], self.fs, self.T_batch, self._rng_state),
            utils.noise_from_sqrt_psd([noise_files[1].strip()], self.fs, self.T_batch, self._rng_state)))

        self.ti = 0                     # index running through input noise batch
        self.linked_components = {}
        self.out = None                 # output value of this components

        if plot_dir:
            plotting.plot_psd(self.readout_noise_tt[0], self.T_fft, self.fs,
                              os.path.join(plot_dir, 'readout_noise_soft_spectrum'),
                              ylabel='Readout noise, soft mode [rad/Hz$^{1/2}$]')
            plotting.plot_psd(self.readout_noise_tt[1], self.T_fft, self.fs,
                              os.path.join(plot_dir, 'readout_noise_hard_spectrum'),
                              ylabel='Readout noise, hard mode [rad/Hz$^{1/2}$]')

    def step(self, dof):
        self.out = eval(self.matrix) @ dof + self.readout_noise_tt[:, self.ti]
        self.ti += 1

    def reset_counters(self):
        self.ti = 0

    def substitute_names_by_variables(self, comp):
        if isinstance(self.matrix, str) and self.matrix.find(comp.name) != -1:
            self.linked_components[comp.name] = comp
            self.matrix = self.matrix.replace(comp.name, 'self.linked_components[\''+comp.name+'\']')
        elif not isinstance(self.matrix, str) and len(self.matrix) > 1:
            for k1 in range(len(self.matrix)):
                for k2 in range(len(self.matrix[k1])):
                    if isinstance(self.matrix[k1][k2], str) and self.matrix[k1][k2].find(comp.name) != -1:
                        self.linked_components[comp.name] = comp
                        self.matrix[k1][k2] = self.matrix[k1][k2].replace(comp.name, 'self.linked_components[\''+comp.name+'\']')


class Controller:
    """
    to-do: change so that it can be an arbitrary MIMO controller (only works for NxN MIMO at the moment)
    """
    def __init__(self, config_data, seed=None, plot_dir=False):
        self._rng_state = np.random.RandomState(seed=seed)
        utils.log_attributes(self, dict(seed=seed))

        self.name = config_data['name']                                     # name of controller
        self.type = config_data['type']                                     # type of component (Sensor)
        self.fs = eval(str(config_data['simulation_sampling_frequency']))   # sampling frequency [Hz]
        self.T_batch = eval(str(config_data['duration_batch']))             # duration of simulated batch [s]
        self.matrix = config_data['matrix']                                 # actuation matrix

        # control filter as second-order sections
        self.controller_sos = []
        self.controller_sos_state = []

        self.actuation = np.zeros((2, 1))

        self.set_sos_models(config_data, plot_dir)
        self.act_point = 'sus' if 'SUS' in config_data['output'] else 'mirror'
        self.ti = 0                                                 # index running through input noise batch
        self.linked_components = {}
        self.out = None                                             # output value of this components

    def set_sos_models(self, config_data, plot_dir):
        zz = np.array([eval(str(comp)) for comp in config_data['filter'][0][0]])
        pp = np.array([eval(str(comp)) for comp in config_data['filter'][0][1]])
        k = eval(str(config_data['filter'][0][2][0]))

        zpk = signal.bilinear_zpk(zz, pp, k, self.fs)
        soft_sos = signal.zpk2sos(*zpk)

        zz = np.array([eval(str(comp)) for comp in config_data['filter'][1][0]])
        pp = np.array([eval(str(comp)) for comp in config_data['filter'][1][1]])
        k = eval(str(config_data['filter'][1][2][0]))
        zpk = signal.bilinear_zpk(zz, pp, k, self.fs)
        hard_sos = signal.zpk2sos(*zpk)

        self.controller_sos = [soft_sos, hard_sos]
        self.controller_sos_state = [np.zeros((len(soft_sos[:, 0]), 2)), np.zeros((len(hard_sos[:, 0]), 2))]

        if plot_dir:
            plotting.sos_freq_resp(soft_sos, self.fs, os.path.join(plot_dir, 'bode_'+self.name+'_soft.png'))
            plotting.sos_freq_resp(hard_sos, self.fs, os.path.join(plot_dir, 'bode_'+self.name+'_hard.png'))

    def step(self, control_in):
        for i in range(len(control_in)):
            output, zf = utils.faster_sosfilt(self.controller_sos[i], np.array([control_in[i]]), zi=self.controller_sos_state[i])
            self.controller_sos_state[i] = zf
            self.actuation[i] = output[0]

        self.ti += 1

        if isinstance(self.matrix, str):
            matrix_num = eval(str(self.matrix))
        else:
            matrix_num = np.zeros_like(self.matrix, dtype=float)
            for k1 in range(len(self.matrix)):
                for k2 in range(len(self.matrix[k1])):
                    matrix_num[k1, k2] = eval(str(self.matrix[k1][k2]))

        self.out = -(matrix_num @ self.actuation).flatten()

    def reset_counters(self):
        self.ti = 0

    def substitute_names_by_variables(self, comp):
        if isinstance(self.matrix, str) and self.matrix.find(comp.name) != -1:
            self.linked_components[comp.name] = comp
            self.matrix = self.matrix.replace(comp.name, 'self.linked_components[\'' + comp.name + '\']')
        elif not isinstance(self.matrix, str) and len(self.matrix) > 1:
            for k1 in range(len(self.matrix)):
                for k2 in range(len(self.matrix[k1])):
                    if isinstance(self.matrix[k1][k2], str) and self.matrix[k1][k2].find(comp.name) != -1:
                        self.linked_components[comp.name] = comp
                        self.matrix[k1][k2] = self.matrix[k1][k2].replace(comp.name,
                                                                          'self.linked_components[\'' + comp.name + '\']')


def run(system, simulation):
    system.reset_counters()

    n_samples = eval(str(simulation['duration_batch']))*eval(str(simulation['simulation_sampling_frequency']))

    inputs = {'pitch': np.zeros((2, )), 'in_power': 0., 'readout': np.zeros((2, )), 'rad_torque': np.zeros((2, )),
              'act_mirror': np.zeros((2, )), 'act_sus': np.zeros((2, ))}

    cavity_power_tt = np.zeros((n_samples, ))
    actuation_tt = np.zeros((n_samples, 2))
    pitch_tt = np.zeros((n_samples, 2))
    beam_spot_tt = np.zeros((n_samples, 2))
    readout_tt = np.zeros((n_samples, 2))

    beam = system.components[2]
    for k in tqdm.tqdm(range(n_samples-1), disable=not sys.stdout.isatty()):

        outputs = system.step(inputs)

        cavity_power_tt[k] = beam.P
        beam_spot_tt[k, :] = beam.BS
        pitch_tt[k, :] = outputs['pitch']
        readout_tt[k, :] = outputs['readout']
        actuation_tt[k, :] = outputs['act_sus']

        inputs = outputs.copy()

        if simulation['loop'] == 'open':
            inputs['act_sus'] = np.zeros((2,))
            inputs['act_mirror'] = np.zeros((2,))

        if np.all(np.abs(outputs['pitch'][0]) > 1):
            print('Diverging time series at', np.round(100.*k/n_samples), '%')
            sys.exit(0)

    return [[pitch_tt, "rad", "pitch", "Pitch (ITM/ETM)"], [cavity_power_tt, "W", "power", "Cavity power"],
            [actuation_tt, "Nm", "actuation", "Control output (ITM/ETM)"],
            [beam_spot_tt, "m", "beam_spot", "Beam-spot motion (ITM/ETM)"],
            [readout_tt, "rad", "readout", "Readout (soft/hard)"]]
