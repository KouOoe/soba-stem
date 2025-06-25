import hyperspy.api as hs
import numpy as np
import os, yaml

from .base import STEMPixelatedDetectorDataLoader
from smatrix_optim import utility


class EMPADLoader(STEMPixelatedDetectorDataLoader):
    
    def load_data(self):

        """
        Load a 4D STEM dataset from EMPAD.

        Expects the configuration to have:
          - loader_params: {'data_file': 'path/to/data.zarr', ...}
          - wavelength, voltage, aperture information.
        """
        empad_dir  = os.path.dirname(self.config_path)
        empad_path = empad_dir +"/"+ os.path.basename(empad_dir) + ".xml"
        if empad_path is None:
            raise ValueError("EMPAD data file path not provided in config.")
        
        with open(self.config_path, 'r') as file:
            config = yaml.safe_load(file)
    
        wavelength = utility.wavelength(config["voltage"])
            
        # Load the EMPAD data using Hyperspy

        empad_data = np.array((hs.load(empad_path, reader='EMPAD')).data, dtype=np.float32)
        empad_data = np.transpose(empad_data, (0, 1, 3, 2))  # Transpose to match the expected shape

        self.raw_params = {
            "data": empad_data,
            "voltage": config["voltage"],
            "aperture": config["aperture"],
            "sampling_scan": config["sampling_scan"],
            "sampling_diff": [config["sampling_diff"][0]*1e-3/wavelength, config["sampling_diff"][1]*1e-3/wavelength],
            "rot_offset_deg": config["rot_offset_deg"],
        }

        # Map parameters to the standard format
        mapped_params = self.map_parameters(self.raw_params)
        return mapped_params