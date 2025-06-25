import numpy as np
import os, yaml, math
import h5py
from omegaconf import OmegaConf

from .base import STEMPixelatedDetectorDataLoader
from smatrix_optim import utility

sim_precision = 'single'

class MustemLoader(STEMPixelatedDetectorDataLoader):

    #def __init__(self, binfactor=1):
    #    self.bifactor = binfactor

    def load_data(self):
        
        dirpath  = os.path.dirname(self.config_path)

        with open(self.config_path, 'r') as inputfile:
            inline = inputfile.readline()

            probe_menu_choice_default = True

            while inline:
                if 'Output filename' in inline:
                    prefix = inputfile.readline().strip()
                    
                if 'Input crystal file name' in inline:
                    crystal_file = inputfile.readline().strip()
                    
                if 'Probe accelerating voltage (kV)' in inline:
                    voltage = float(inputfile.readline().strip())
                    
                if 'Thickness' in inline:
                    thicknesses = list(map(float, inputfile.readline().strip().split(':')))
                    
                if 'Tile supercell x' in inline:
                    tile_x = int(inputfile.readline().strip())
                    
                if 'Tile supercell y' in inline:
                    tile_y = int(inputfile.readline().strip())
                    
                if 'Number of pixels in x' in inline:
                    supercell_nx = int(inputfile.readline().strip())
                    
                if 'Number of pixels in y' in inline:
                    supercell_ny = int(inputfile.readline().strip())
                    
                if 'aperture cutoff' in inline:
                    aperture = float(inputfile.readline().strip()) # in mrad
                    
                if 'Defocus' in inline:
                    defocus = float(inputfile.readline().strip())
                    
                if '<1> QEP <2> ABS' in inline:
                    QEP_ABS_select = int(inputfile.readline().strip())
                    
                if 'diffraction pattern y pixels' in inline:
                    nky = int(inputfile.readline().strip())
                    CBED_crop_choice = True
                    
                if 'diffraction pattern x pixels' in inline:
                    nkx = int(inputfile.readline().strip())
                    CBED_crop_choice = True

                if 'nxsample' in inline:
                    nx = int(inputfile.readline().strip())
                    probe_menu_choice_default = False

                if 'nysample' in inline:
                    ny = int(inputfile.readline().strip())
                    probe_menu_choice_default = False

                inline = inputfile.readline()

        with open(dirpath+'/'+crystal_file, 'r') as crystal:
            crystal.readline()
            lattice_const = list(map(float, crystal.readline().strip().split()))
            n_atom = list(map(int, crystal.readline().strip().split()))
        
        rx, ry, rz = lattice_const[0], lattice_const[1], lattice_const[2]

        if probe_menu_choice_default==True:
            # return Nyquist sampling condition
            k0 = aperture*1e-3/utility.wavelength(voltage)
            nx = math.ceil(rx * 4 * k0)
            ny = math.ceil(ry * 4 * k0)
        else:
            pass

        dkx = 1/rx/tile_x
        dky = 1/ry/tile_y
        drx = rx / nx
        dry = ry / ny

        def return_thickness(self, thickness):
            n_slice = math.floor(thickness/self.rz)

            if abs(thickness - self.rz*n_slice) < abs(thickness - self.rz*(n_slice+1)):
                pass
            else:
                n_slice = n_slice+1
            
            return int(n_slice * self.rz)
    
        # QEP or absroption?
        if QEP_ABS_select == 2:
            cbed_prefix = 'abs_Diffraction_pattern'
        elif QEP_ABS_select == 1:
            cbed_prefix = 'Diffraction_pattern'

        # Thickness, signle or serial?
        if len(thicknesses) == 1:
            thickness_prefix = ''
        else:
            thickness_prefix = '_z={0}_A'.format(return_thickness(self.thickness_simulated_mustem))


        if sim_precision == 'single':
            calc_precision = '>f4'
        elif sim_precision == 'double':
            calc_precision = '>f8'

        #cbed = np.zeros(nkx//self.binfactor*nky//self.binfactor*nx*ny)
        cbed = np.zeros((ny, nx, nky//self.binfactor, nkx//self.binfactor), dtype=np.float32)
        for iy in range(ny):
            for ix in range(nx):
                cbed_tmp = np.fromfile(dirpath + '/' + prefix + thickness_prefix + '_pp_{0}_{1}_'.format(ix+1, iy+1) + cbed_prefix + '_{0}x{1}.bin'.format(nkx, nky), dtype=calc_precision).reshape((nky, nkx))
                #print(np.shape(cbed_tmp))
                cbed_tmp = bin_image(image=cbed_tmp, bin_size=self.binfactor, mode='sum')
                #cbed[nkx//self.binfactor*nky//self.binfactor*(ix+iy*nx):nkx//self.binfactor*nky//self.binfactor*(ix+iy*nx+1)] = cbed_tmp
                cbed[iy,ix] = cbed_tmp

        #cbed = cbed.reshape(ny, nx, nky, nkx).astype(np.float32)
        #cbed = cbed/(self.dky*self.dkx)


        self.raw_params = {
            "data": np.array(cbed, dtype=np.float32),
            "voltage": voltage,
            "aperture": aperture,
            "sampling_scan": [dry, drx],
            "sampling_diff": [dkx, dky],
            "rot_offset_deg": 0,
            "thickness": thicknesses,
            "defocus": defocus,
        }





def bin_image(image: np.ndarray,
              bin_size: int | tuple[int, int],
              mode: str = 'sum') -> np.ndarray:
    """
    2D 画像配列をビニングする関数。

    Parameters
    ----------
    image : np.ndarray
        入力画像（2D配列）。
    bin_size : int or tuple of int
        ビニングサイズ。int を指定すると縦横同一サイズ、
        (bin_y, bin_x) を指定するとそれぞれの方向のサイズ。
    mode : {'sum', 'mean'}
        ピクセル強度の統合方法。'sum'=総和, 'mean'＝平均。

    Returns
    -------
    binned : np.ndarray
        ビニング後の画像。
    """
    # bin_size をタプル化
    if isinstance(bin_size, int):
        bin_y = bin_x = bin_size
    else:
        bin_y, bin_x = bin_size

    H, W = image.shape
    H_crop = (H // bin_y) * bin_y
    W_crop = (W // bin_x) * bin_x

    # 余分な行列要素を切り捨て
    img_crop = image[:H_crop, :W_crop]

    # (H_crop//bin_y, bin_y, W_crop//bin_x, bin_x) にリシェイプ
    img_reshaped = img_crop.reshape(
        H_crop // bin_y, bin_y,
        W_crop // bin_x, bin_x
    )

    # ビニング（軸1と3をまとめて sum/mean）
    if mode == 'sum':
        binned = img_reshaped.sum(axis=(1, 3))
    elif mode == 'mean':
        binned = img_reshaped.mean(axis=(1, 3))
    else:
        raise ValueError("mode は 'sum' か 'mean' のいずれかを指定してください。")

    return binned
