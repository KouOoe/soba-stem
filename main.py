import os, yaml, argparse
import numpy as np
import torch
import scipy.ndimage as ndimage
from Pixel_class import Pixelated_STEM
from collections import defaultdict
from omegaconf import OmegaConf

from smatrix_optim.data_loader.empad_loader import EMPADLoader
from smatrix_optim.data_loader.smatrix_calc_cbed_loader import SmatrixCalcCBEDLoader
from smatrix_optim.data_loader.abtem_yaml_loader import AbTEMLoader
from smatrix_optim.data_loader.mustem_loader import MustemLoader
from smatrix_optim.utility import wavelength, interpolate

from SMatrixClassTorch import SMatrix as SMatrixTorch
from SMatrixClass import SMatrix as SMatrixNumpy

import make_potential_class


def parse_args():
    parser = argparse.ArgumentParser(description="コマンドライン引数で指定されたYAMLファイルから設定を読み込む")
    # --yamlオプションを追加
    parser.add_argument("--yaml", type=str, required=True, help="YAML設定ファイルのパス")
    return parser.parse_args()

def load_yaml_file(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
        return defaultdict(lambda: None, config) #yaml.safe_load(file)
    

def main():
    #################################################################
    # read yaml config file and load data
    #################################################################
    args         = parse_args()
    config_optim = OmegaConf.load(args.yaml) #load_yaml_file(args.yaml)

    if config_optim['data_type'] == "empad":
        data_loader = EMPADLoader(config_optim['config_path'])
    elif config_optim['data_type'] == "smatrix_calc_cbed":
        data_loader = SmatrixCalcCBEDLoader(config_optim['config_path'])
    elif config_optim['data_type'] == "abtem":
        data_loader = AbTEMLoader(config_optim['config_path'])
    elif config_optim['data_type'] == "mustem":
        data_loader = MustemLoader(config_optim['config_path'])
    else:
        raise ValueError("Invalid data type specified in the configuration.")
    
    # Load the data
    data_loader.load_data()
    cbed = data_loader.raw_params['data']
    cbed = data_loader.binning_diff(cbed, config_optim['binning_diff'])

    pixel = Pixelated_STEM(
        voltage =data_loader.raw_params['voltage'],
        aperture=data_loader.raw_params['aperture'],
    )
    pixel.load_parameter(
        thickness=config_optim['thickness_init'],
        defocus  =config_optim['defocus_init'],
        dry=data_loader.raw_params['sampling_scan'][0],
        drx=data_loader.raw_params['sampling_scan'][1],
        dkx=data_loader.raw_params['sampling_diff'][0]*config_optim['binning_diff'],
        dky=data_loader.raw_params['sampling_diff'][1]*config_optim['binning_diff'],
        rot_offset_rad=data_loader.raw_params['rot_offset_deg']*np.pi/180,
    )
    pixel.cbed_import(cbed/np.sum(cbed, axis=(2,3))[:,:,np.newaxis,np.newaxis])
    #print(data_loader.raw_params)

    obf_from_measured_cbed = pixel.OBF_pixel()


    ##################################################################
    # make a save directory
    ##################################################################

    save_dir = os.path.dirname(args.yaml) + "/" + os.path.basename(args.yaml).split(".")[0]
    os.makedirs(save_dir, exist_ok=True)
    save_dir+= "/"


    ##################################################################
    # calc IAM potential from xtl file
    ##################################################################

    if "xtl_file_path" in config_optim:
        nry_iam_pot_uc = int(pixel.ry/config_optim['unitcell_tiling'][0]/config_optim['potential_grid_sampling'])
        nrx_iam_pot_uc = int(pixel.rx/config_optim['unitcell_tiling'][1]/config_optim['potential_grid_sampling'])

        nry_iam_pot_sc = nry_iam_pot_uc*config_optim['unitcell_tiling'][0]
        nrx_iam_pot_sc = nrx_iam_pot_uc*config_optim['unitcell_tiling'][1]

        iam_pot = make_potential_class.makepot(
            filepath=config_optim['xtl_file_path'],
            voltage_kV=config_optim['voltage'],
            ngrid=[nrx_iam_pot_uc, nry_iam_pot_uc],
        )

        Ug_els_uc = iam_pot.optical_pot_Ug(absorption=False)
        Ug_els_sc = np.fft.fft2(np.tile(np.fft.ifft2(Ug_els_uc), (config_optim['unitcell_tiling'][0], config_optim['unitcell_tiling'][1])))
        Ug_els_sc = Ug_els_sc/config_optim['unitcell_tiling'][0]/config_optim['unitcell_tiling'][1]

        obf_potential_grid = interpolate(
            obf_from_measured_cbed,
            nry_iam_pot_sc,
            nrx_iam_pot_sc,
        )

    else:
        obf_potential_grid = interpolate(
            obf_from_measured_cbed,
            int(pixel.nry*pixel.dry/config_optim['potential_grid_sampling']),
            int(pixel.nrx*pixel.drx/config_optim['potential_grid_sampling']),
        )

    Ug_from_OBF = np.fft.fft2(obf_potential_grid/pixel.wavelength/np.pi)\
                    *config_optim['potential_grid_sampling']*config_optim['potential_grid_sampling']/pixel.ry/pixel.rx/config_optim.thickness



    #######################################################################
    # Set up the S-matrix beam condition
    #######################################################################

    # set beams
    smatrix_np = SMatrixNumpy(
        voltage       = pixel.voltage,
        aperture_mrad = pixel.aperture,
    )
    smatrix_np.make_grid(
        ry = pixel.ry,
        rx = pixel.rx,
        ny = Ug_from_OBF.shape[0],
        nx = Ug_from_OBF.shape[1],
    )
    smatrix_np.set_beams_from_image(image_fft=Ug_from_OBF, amp_cutoff=0e-10, show_beams=False, cutoff_k_frac=config_optim["beam_outer_angle"]*1.01)

    #######################################################################
    # Set up the pytorch S-matrix calculation
    #######################################################################

    # gpu or cpu for optimization
    if torch.cuda.is_available():
        device  = torch.device('cuda')
        use_gpu = True
    else:
        device = torch.device('cpu')

    if config_optim['use_gpu']:
        device  = torch.device('cpu')
        use_gpu = False

    # --- Parameters to be optimized --- #
    thickness   = torch.tensor(config_optim["thickness_init"], device=device, requires_grad=config_optim["optim_thickness"], dtype=torch.float32)
    defocus     = torch.tensor(config_optim["defocus_init"], device=device, requires_grad=config_optim["optim_defocus"], dtype=torch.float32)
    source_fwhm = torch.tensor(config_optim["sourcesize_init"], device=device, requires_grad=config_optim["optim_sourcesize"], dtype=torch.float32)

    smatrix_torch = SMatrixTorch(
        voltage       = pixel.voltage,
        aperture_mrad = pixel.aperture,
        gpu_use       = use_gpu
    )
    smatrix_torch.make_grid(
        ry = pixel.ry,
        rx = pixel.rx,
        ny = Ug_from_OBF.shape[0],
        nx = Ug_from_OBF.shape[1],
    )
    smatrix_torch.import_beams(
        beam_coordinates=smatrix_np.beam_coordinates,
        beam_indices=smatrix_np.beam_indices
    )

    # optimisation loop using optimisation.py



if __name__ == "__main__":
    main()