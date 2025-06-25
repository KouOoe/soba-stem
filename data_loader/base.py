# stem_opt/data_loader/base.py
from abc import ABC, abstractmethod
import numpy as np
import scipy.ndimage as ndimage

class STEMPixelatedDetectorDataLoader(ABC):
    def __init__(self, config_path):
        """
        Initialize the loader with a configuration dictionary.
        :param config: A dictionary or configuration object containing loader settings.
        """
        self.config_path = config_path

    @abstractmethod
    def load_data(self):
        """
        Load the 4D STEM dataset and its associated parameters.

        Returns a dictionary containing at least:
          - data: The 4D STEM dataset (e.g., numpy array or torch tensor)
          - detector: Detector configuration (e.g., type, geometry)
          - wavelength: Electron wavelength
          - voltage: Accelerating voltage
          - aperture: Aperture size or a related parameter

        Subclasses must override this method.
        """
        pass

    def map_parameters(self, raw_params):
        """
        Optionally map raw parameters from the loader-specific format to the
        standard format required by the optimization code.
        :param raw_params: A dict with raw parameters.
        :return: A dict with standardized keys.
        """
        mapped = {
            "data": raw_params.get("data"),
            "voltage": raw_params.get("voltage"),
            "aperture": raw_params.get("aperture"),
            "sampling_scan": raw_params.get("sampling_scan"),
            "sampling_diff": raw_params.get("sampling_diff"),
        }
        return mapped
    
    def binning_diff(self, data, bin_factor):
        """
        Apply binning to the data in the diffraction dimension.
        :param data: The 4D STEM dataset.
        :param bin_factor: The factor by which to bin the data.
        :return: Binned data.
        """
        binned_data = data.reshape(data.shape[0], data.shape[1],
                                   data.shape[2] // bin_factor, bin_factor,
                                   data.shape[3] // bin_factor, bin_factor).mean(axis=(3, 5))
        return binned_data
    
    def centering_crop_diff_com(self, data, crop_size):
        """
        Center crop the data in the diffraction dimension.
        :param data: The 4D STEM dataset.
        :param crop_size: The size of the crop.
        :return: Cropped data.
        """
        center       = ndimage.center_of_mass(np.sum(data, axis=(0, 1)))
        cropped_data = data[:,:,int(center[0]-crop_size[0]/2):int(center[0]-crop_size[0]/2)+crop_size[0],
                            int(center[1]-crop_size[1]/2):int(center[1]-crop_size[1]/2)+crop_size[1]]
                            #round(center[0]-crop_size[0]/2):round(center[0]+crop_size[0]/2),
                            #ound(center[1]-crop_size[1]/2):round(center[1]+crop_size[1]/2)]
        return cropped_data
    
    def measure_rot_offset(self, data, rot_step_deg=15):
        """
        Measure the rotation offset of the data.
        :param data: The 4D STEM dataset.
        :param rot_step: The step size for rotation.
        :return: The measured rotation offset.
        """
        rot_ary  = np.arange(0,360,rot_step_deg)
        curl_ary = np.zeros(len(rot_ary))

        return 0