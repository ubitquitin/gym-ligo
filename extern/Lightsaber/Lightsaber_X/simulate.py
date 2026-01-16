"""Main simulator runner for LIGO Lightsaber."""

import json
import os
import time
import pathlib

from absl import app
import fancyflags as ff
import numpy as np
import yaml

from Lightsaber import *
import utils
import plotting

import pathlib
from typing import Any, Dict, Optional, Union

PathLike = Union[str, pathlib.Path]


def save_results_as_txt(results, simulation, directory):
    """
        Saves the results as txt files and plots of the spectral densities of the time series.

        Arguments:
        results: The dataclass that contains the results: time series, unit, name.
        directory: Folder to write the data into.
    """

    fs = eval(str(simulation['simulation_sampling_frequency']))
    t_fft = eval(str(simulation['duration_fft']))

    directory = pathlib.Path(directory)
    for result in results:
        filename = os.path.join(directory, f'{result[2]}.csv')
        np.savetxt(filename, result[0], delimiter=' ')

        filename = os.path.join(directory, f'{result[2]}.png')
        plotting.plot_psd(result[0], t_fft, fs, filename, ylabel=result[3]+' ['+result[1]+'/Hz$^{1/2}$]')


def create_system(config, simulation, plot_dir=False):
    """
    At the moment, this function is a dirty hack to work with the arm cavity model. It will be substituted by a
    function that parses nodes and connects components automatically.
    """
    seed = eval(str(simulation['seed_for_random']))

    system = System()

    for comp in config.keys():
        config[comp]['name'] = comp
        config[comp]['simulation_sampling_frequency'] = simulation['simulation_sampling_frequency']
        config[comp]['duration_batch'] = simulation['duration_batch']
        config[comp]['duration_fft'] = simulation['duration_fft']
        system.append(eval(config[comp]['type'])(config[comp], seed, plot_dir))

    for beam in system.get_by_type('Beam'):
        mirrors = system.get_by_type('Mirror')
        laser = system.get_by_type('Laser')
        beam.set_parameters(mirrors[0], mirrors[1], laser[0].wavelength)

    return system


def link_components(system):
    for comp in system.components:
        for comp_to_be_linked in system.components:
            comp_to_be_linked.substitute_names_by_variables(comp)


def main(argv):
    del argv  # Unused.

    with open('configuration/config.yaml') as f:
        #config = yaml.load(f, Loader=yaml.loader.BaseLoader)
        config = yaml.load(f, Loader=yaml.FullLoader)

    simulation = []
    key_to_delete = []
    for key in config.keys():
        if config[key]['type'] == 'Simulation':
            simulation = config[key]
            key_to_delete = key
    del config[key_to_delete]

    output = []
    for key in config.keys():
        if config[key]['type'] == 'Output':
            output = config[key]
            key_to_delete = key
    del config[key_to_delete]

    system = create_system(config, simulation, output['out_directory'])
    link_components(system)

    results = run(system, simulation)

    output_fs = eval(str(output['output_sampling_frequency']))
    fs = eval(str(simulation['simulation_sampling_frequency']))
    if output_fs < fs:
        results = utils.decimate(results, fs, output_fs)
        simulation['simulation_sampling_frequency'] = output['output_sampling_frequency']

    save_results_as_txt(results, simulation, output['out_directory'])


if __name__ == '__main__':
    start_time = time.time()
    app.run(main)
    print("--- %s seconds ---" % np.round(time.time() - start_time))
