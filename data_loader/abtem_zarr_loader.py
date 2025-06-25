import numpy as np
import os
from omegaconf import OmegaConf

from .base import STEMPixelatedDetectorDataLoader
from smatrix_optim import utility
import zarr

class AbTEMZarrLoader(STEMPixelatedDetectorDataLoader):

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

        path_pixel = data_dir+'/'+exp_name+'_pixel.zarr'
        pixel_data = zarr.open(path_pixel, 'r')
        abtem_data = pixel_data['array0']

        voltage    = config["energy"] * 1e-3
        wavelength = utility.wavelength(voltage)
        aperture   = config["semiangle_cutoff"]

        self.raw_params = {
            "data": np.array(abtem_data, dtype=np.float32),
            "voltage": voltage,
            "aperture": aperture,
            "sampling_scan": [pixel_data.attrs.get("kwargs0")['ensemble_axes_metadata'][1]["sampling"],
                              pixel_data.attrs.get("kwargs0")['ensemble_axes_metadata'][0]["sampling"]],
            "sampling_diff": [pixel_data.attrs.get("kwargs0")['sampling'][1],
                              pixel_data.attrs.get("kwargs0")['sampling'][0]],
            "rot_offset_deg": 0
        }
        # Map parameters to the standard format
        mapped_params = self.map_parameters(self.raw_params)
        return mapped_params