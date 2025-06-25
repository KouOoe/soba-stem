import numpy as np
import os, yaml
import h5py
from omegaconf import OmegaConf

from .base import STEMPixelatedDetectorDataLoader
from smatrix_optim import utility


class AbTEMLoader(STEMPixelatedDetectorDataLoader):

    def load_data(self):
        """
        Load a 4D STEM dataset from AbTEM.
        Expects the configuration to have:
            - loader_params: {'data_file': 'path/to/data.hdf5', ...}
            - wavelength, voltage, aperture information.
        """

        config = OmegaConf.load(self.config_path)

        exp_name = os.path.splitext(os.path.basename(self.config_path))[0]
        data_dir = self.config_path.replace('.yaml','')

        abtem_data = h5py.File(data_dir+"/"+exp_name+"_pixel.hdf5", 'r')

        voltage    = config["energy"] * 1e-3
        wavelength = utility.wavelength(voltage)
        aperture   = config["semiangle_cutoff"]

        self.raw_params = {
            "data": np.array(abtem_data['array'], dtype=np.float32),
            "voltage": voltage,
            "aperture": aperture,
            "sampling_scan": [abtem_data['sampling'][1], abtem_data['sampling'][0]],
            "sampling_diff": [abtem_data['sampling'][3]*1e-3/wavelength, abtem_data['sampling'][2]*1e-3/wavelength],
            "rot_offset_deg": 0
        }
        # Map parameters to the standard format
        mapped_params = self.map_parameters(self.raw_params)
        return mapped_params