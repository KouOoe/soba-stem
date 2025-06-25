import numpy as np
import os, h5py

from .base import STEMPixelatedDetectorDataLoader
from smatrix_optim import utility


class SmatrixCalcCBEDLoader(STEMPixelatedDetectorDataLoader):

    def load_data(self):
        """
        Load a 4D STEM dataset from AbTEM.
        Expects the configuration to have:
            - loader_params: {'data_file': 'path/to/data.hdf5', ...}
            - wavelength, voltage, aperture information.
        """

        with h5py.File(self.config_path, 'r') as f:
            pixel_data_smatrix = f['cbed'][:,:,:,:]
            voltage            = f['voltage'][()]
            aperture_cutoff    = f['aperture'][()]
            thickness  = f['thickness'][()]
            defocus    = f['defocus'][()]
            fov        = f['FOV'][()]
            sampling   = f['sampling'][()]



        self.raw_params = {
            "data": pixel_data_smatrix.astype(np.float32),
            "voltage": voltage,
            "aperture": aperture_cutoff,
            "sampling_scan": [sampling[0], sampling[1]],
            "sampling_diff": [sampling[2], sampling[3]],
            "rot_offset_deg": 0,
        }
        # Map parameters to the standard format
        mapped_params = self.map_parameters(self.raw_params)
        return mapped_params