import numpy as np
from scipy import ndimage
from scipy.optimize import minimize

from matplotlib import pyplot as plt, scale
import sys, math

import torch
import torch.nn.functional as F


# constant
hc   = 12.398                # keV*Å
m0c2 = 511.                  # keV
e    = 1.6022e-19            # C
pi_tensor = torch.acos(torch.zeros(1, device='cpu')).item()*2



class SMatrixFwdCalc():

    def __init__(self, voltage, aperture_mrad, gpu_use=True):
        super().__init__()

        self.voltage    = voltage
        self.aperture   = aperture_mrad

        self.wavelength = hc / np.sqrt( self.voltage * ( 2 * m0c2 + self.voltage ) ) # Å
        self.k0         = self.aperture * 1e-3 / self.wavelength                     # Å-1
        self.zprobe     = 1 / self.k0**2 / self.wavelength

        self.device_kind = torch.device("cuda" if (torch.cuda.is_available() & gpu_use) else "cpu")
        self.k0          = torch.tensor(self.k0, device=self.device_kind)
        self.wavelength  = torch.tensor(self.wavelength, device=self.device_kind)


        
    def make_grid(self, ry, rx, ny, nx):
        self.ry  = torch.tensor(ry, device=self.device_kind)
        self.rx  = torch.tensor(rx, device=self.device_kind)
        self.nry = torch.tensor(ny, device=self.device_kind)
        self.nrx = torch.tensor(nx, device=self.device_kind)
        self.dry = ry/ny
        self.drx = rx/nx
        self.dqy = torch.tensor(1/ry, device=self.device_kind)
        self.dqx = torch.tensor(1/rx, device=self.device_kind)

        self.rgrid = torch.meshgrid(torch.arange(0, ry, self.dry, device=self.device_kind), torch.arange(0, rx, self.drx, device=self.device_kind), indexing='ij')
        self.qgrid = torch.meshgrid(torch.fft.fftfreq(self.nry, self.dry, device=self.device_kind), torch.fft.fftfreq(self.nrx, self.drx, device=self.device_kind), indexing='ij')


    def import_beams(self, beam_coordinates, beam_indices):
        self.n_beam           = len(beam_indices[0])
        self.beam_coordinates = torch.tensor(np.array(beam_coordinates), device=self.device_kind)
        self.beam_indices     = torch.tensor(np.array(beam_indices), device=self.device_kind)

    

    def make_Amatrix_from_pot_vector(self, Ug_pot):
        """
        Construct the A–matrix from the input potential vector.
        Off–diagonal elements are computed using the differences in beam indices.
        """
        Ug_els, Ug_abs = Ug_pot  # unpack elastic and absorptive parts
        # Compute index differences (vectorized)
        diff_indices_x = self.beam_indices[0][:, None] - self.beam_indices[0]
        diff_indices_y = self.beam_indices[1][:, None] - self.beam_indices[1]
        
        # Compute the off-diagonal elements via vectorized indexing
        A_matrix = Ug_els[diff_indices_x, diff_indices_y] + 1j * Ug_abs[diff_indices_x, diff_indices_y]
        
        # Compute diagonal: beam squared (kinetic term) plus absorptive part at (0,0)
        beam_sq = self.beam_coordinates[0]**2 + self.beam_coordinates[1]**2
        diagonal_values = -beam_sq + 1j * Ug_abs[0, 0]
        
        # Zero out the current diagonal and add the computed diagonal
        A_matrix.fill_diagonal_(0)
        return A_matrix + torch.diag(diagonal_values)


    def expm_pade_scaled_torch_optimized(self, A, m):

        device = A.device
        dtype = A.dtype
        n = A.size(0)
        c = torch.tensor(np.array([1, 1/2, 1/6, 1/24, 1/120, 1/720, 1/5040, 1/40320]), dtype=dtype, device=device)
    
        A_scaled = A / m
        
        powers = [torch.eye(n, dtype=dtype, device=device)]
        for i in range(1, 8):
            powers.append(torch.matmul(powers[-1], A_scaled))

        
        U = sum(ci * pi for ci, pi in zip(c, powers))
        V = sum(ci * (-pi if i % 2 else pi) for i, (ci, pi) in enumerate(zip(c, powers)))
        
        R = torch.matmul(torch.inverse(V), U)
        expA = torch.matrix_power(R, m)
        
        return expA


    
    def calc_Smatrix_from_Amatrix_Pade(self, Amatrix, thickness, scale_factor=2):
        """
        Calculate the scattering matrix using a Pade–based matrix exponential.
        Uses torch.pi instead of an external pi_tensor.
        """
        factor = 1j * torch.pi * self.wavelength * thickness
        # Multiply Amatrix by the phase factor and compute the exponential
        return self.expm_pade_scaled_torch_optimized(Amatrix * factor, scale_factor)

    

    def phasefactor_probe(self, ry, rx, ky, kx):
        return torch.exp(-pi_tensor*2j*(ry*ky + rx*kx))

    def phasefactor_probe_vector(self, ry, rx, k):
        # kからky, kxを取り出す
        ky = k[0, :]
        kx = k[1, :]

        # ry, rxから2次元グリッドを生成
        ry_grid, rx_grid = torch.meshgrid(ry, rx, indexing='ij')

        # グリッド上の各点に対して、ky, kxとの位相因子を計算
        # 形状の拡張（unsqueeze）を使用して、ブロードキャスティングを可能にする
        phase_factor = torch.exp(-pi_tensor * 2j * (ry_grid.unsqueeze(-1) * ky + rx_grid.unsqueeze(-1) * kx))

        return phase_factor


    def phasefactor_probe_vector_as_stack(self, ry, rx, k):
        # kからky, kxを取り出す
        ky = k[0, :]
        kx = k[1, :]

        # ry, rxから2次元グリッドを生成
        ry_grid, rx_grid = torch.stack([ry, rx])#, dim=-1)

        # グリッド上の各点に対して、ky, kxとの位相因子を計算
        # 形状の拡張（unsqueeze）を使用して、ブロードキャスティングを可能にする
        phase_factor = torch.exp(-pi_tensor * 2j * (ry_grid.unsqueeze(-1) * ky + rx_grid.unsqueeze(-1) * kx))

        return phase_factor



    def D_torch(self,ky,kx,inner,outer,theta1=-pi_tensor,theta2=pi_tensor,rot_offset=torch.tensor(0)):
        # inner/outer angle is normalized by aperture angle i.e. r=1 means edge of BF disc.
        inner = inner * self.k0
        outer = outer * self.k0
        rot_epsilon = torch.tensor(1e-6)

        kx_rot = kx*torch.cos(rot_offset+rot_epsilon) + ky*torch.sin(rot_offset+rot_epsilon)
        ky_rot =-kx*torch.sin(rot_offset+rot_epsilon) + ky*torch.cos(rot_offset+rot_epsilon)

        mask_radius   = torch.logical_and(ky**2 + kx**2 < outer**2, ky**2 + kx**2 >= inner**2)
        mask_azimuth  = torch.logical_and(torch.atan2(ky_rot, kx_rot) >= theta1, torch.atan2(ky_rot, kx_rot) < theta2)
        mask          = torch.logical_and(mask_radius, mask_azimuth)
        
        return mask


    def A_torch(self,ky,kx):
        aperture = (ky**2 + kx**2 <= self.k0**2)/torch.sqrt(pi_tensor*self.k0**2)
        return aperture

    def A_torch_soft(self,ky,kx):
        soft_aperture = torch.where(ky**2+kx**2>0, (self.k0*torch.sqrt(ky**2+kx**2)-(ky**2+kx**2))/torch.sqrt(((ky*self.dqy)**2+(kx*self.dqx)**2)), torch.ones(ky.size(), device=self.device_kind, dtype=torch.float64))
        soft_aperture = torch.where(soft_aperture+0.5>0,soft_aperture+0.5,torch.zeros(ky.size(), device=self.device_kind, dtype=torch.float64))
        soft_aperture = torch.where(soft_aperture>1,torch.ones(ky.size(), device=self.device_kind, dtype=torch.float64),soft_aperture)
        return soft_aperture/torch.sqrt(pi_tensor*self.k0**2)
    

    def chi_torch(self,ky,kx,defocus=torch.tensor(0),tilt_rad=torch.tensor([0,0]),tilt_z=torch.tensor(0),coma=torch.tensor([0,0]),twofold_stig=torch.tensor([0,0]),threefold_stig=torch.tensor([0,0])):
        # twofold_stig[0] is C_12a and twofold_stig[1] is C_12b
        #pi_tensor * defocus * (self.wavelength * (ky**2 + kx**2) - 2*ky*torch.tan(tilt_rad[0]) - 2*kx*torch.tan(tilt_rad[1]))\
        chi = pi_tensor * defocus * (self.wavelength * (ky**2 + kx**2))\
                - 2*pi_tensor*tilt_z*(ky*torch.tan(tilt_rad[0]) + kx*torch.tan(tilt_rad[1]))\
                + 2/3*pi_tensor*self.wavelength**2*(ky**2 + kx**2)*(coma[0]*ky + coma[1]*kx)\
                + twofold_stig[0]*pi_tensor*self.wavelength*(kx**2-ky**2) + 2*twofold_stig[1]*pi_tensor*self.wavelength*ky*kx \
                + 2/3*pi_tensor*self.wavelength**2*(threefold_stig[0]*ky*(ky**2-3*kx**2) + threefold_stig[1]*kx*(kx**2-3*ky**2))
        return chi


    def T_torch(self,ky,kx,defocus=torch.tensor(0),tilt_rad=torch.tensor([0,0]),tilt_z=torch.tensor(0),coma=torch.tensor([0,0]),twofold_stig=torch.tensor([0,0]),threefold_stig=torch.tensor([0,0])):
        lens_transfer = self.A_torch_soft(ky,kx) * torch.exp(-1j*self.chi_torch(ky,kx,defocus,tilt_rad,tilt_z,coma,twofold_stig,threefold_stig))
        #self.A_torch(ky,kx) * torch.exp(-1j*self.chi_torch(ky,kx,defocus,tilt_rad))
        return lens_transfer


    def calc_CBED_from_Smatrix_vector_from_imported_scan_pos(self, smatrix, chunk_size=16, defocus=0, scan_ary=None,
                                                            kmax_calc_frac=2.5,
                                                            tilt_rad=torch.tensor([0, 0], device='cuda'),
                                                            tilt_z=torch.tensor(0, device='cuda'),
                                                            coma=torch.tensor([0, 0], device='cuda'),
                                                            twofold_stig=torch.tensor([0, 0], device='cuda'),
                                                            threefold_stig=torch.tensor([0, 0], device='cuda'),
                                                            show_progress=True):
        """
        Calculate CBED patterns from a scattering matrix using externally provided scan positions.
        The computation is performed in chunks to efficiently handle large datasets.
        """
        # Determine the size of the CBED diffraction pattern
        ny_cbed = int(self.k0 / self.dqy * kmax_calc_frac * 2)
        nx_cbed = int(self.k0 / self.dqx * kmax_calc_frac * 2)

        ny_scan = scan_ary.shape[0]
        nx_scan = scan_ary.shape[1]

        # Precompute slicing indices for the central CBED region from the q–grid.
        ystart = (self.nry - ny_cbed) // 2
        xstart = (self.nrx - nx_cbed) // 2
        yend = ystart + ny_cbed
        xend = xstart + nx_cbed

        self.ky_cbed = torch.fft.ifftshift(torch.fft.fftshift(self.qgrid[0])[ystart:yend, xstart:xend])
        self.kx_cbed = torch.fft.ifftshift(torch.fft.fftshift(self.qgrid[1])[ystart:yend, xstart:xend])

        # Calculate the number of chunks using integer arithmetic.
        chunk_ny = (ny_scan + chunk_size - 1) // chunk_size
        chunk_nx = (nx_scan + chunk_size - 1) // chunk_size

        # Preallocate the output CBED array.
        cbed = torch.empty(ny_scan, nx_scan, ny_cbed, nx_cbed, dtype=torch.float32, device=self.device_kind)

        # Loop over scan chunks.
        for i in range(chunk_ny):
            for j in range(chunk_nx):
                start_y = i * chunk_size
                end_y = min((i + 1) * chunk_size, ny_scan)
                start_x = j * chunk_size
                end_x = min((j + 1) * chunk_size, nx_scan)

                ny_chunk = end_y - start_y
                nx_chunk = end_x - start_x

                # Extract scan positions for the current chunk.
                rpy_chunk = scan_ary[start_y:end_y, start_x:end_x, 0]
                rpx_chunk = scan_ary[start_y:end_y, start_x:end_x, 1]

                # Compute the phase factor for the current chunk.
                phase_factor = self.T_torch(self.beam_coordinates[0], self.beam_coordinates[1],
                                            defocus, tilt_rad, tilt_z, coma, twofold_stig, threefold_stig) * \
                            self.phasefactor_probe_vector_as_stack(rpy_chunk, rpx_chunk, self.beam_coordinates)

                # Compute scattering amplitude using batched matrix multiplication.
                # phase_factor: (ny_chunk, nx_chunk, n_beam), smatrix: (n_beam, n_beam)
                amplitude = torch.einsum('...h,gh->...g', phase_factor, smatrix) #torch.matmul(phase_factor, smatrix)
                intensity = (torch.abs(amplitude) ** 2).float()  # shape: (ny_chunk, nx_chunk, n_beam)

                # Preallocate a temporary CBED chunk and assign the computed intensities 
                cbed_chunk = torch.zeros(ny_chunk, nx_chunk, ny_cbed, nx_cbed, dtype=torch.float32, device=self.device_kind)
                cbed_chunk[:, :, self.beam_indices[0], self.beam_indices[1]] = intensity

                cbed[start_y:end_y, start_x:end_x, :, :] = cbed_chunk / torch.sum(torch.abs(self.T_torch(self.beam_coordinates[0], self.beam_coordinates[1], defocus))**2)

                if show_progress:
                    progress = ((i * chunk_nx + j + 1) / (chunk_ny * chunk_nx)) * 100
                    sys.stdout.write('\r{:.1f}% done...'.format(progress))
                    sys.stdout.flush()

        return cbed
    

    def source_blur_cbed_GandL(self, cbed, n_scan, source_fwhm_G, source_fwhm_L, ratio_GtoL=1, axis=(0,1)):

        ny_scan = n_scan[0]
        nx_scan = n_scan[1]
        rpy_ary = torch.fft.fftfreq(ny_scan, self.ry/ny_scan, device=self.device_kind) #torch.arange(0, self.ry, dr_scan)
        rpx_ary = torch.fft.fftfreq(nx_scan, self.rx/nx_scan, device=self.device_kind) #torch.arange(0, self.rx, dr_scan)
        sgrid   = torch.meshgrid(rpy_ary, rpx_ary, indexing='ij')

        source_sigma_G = source_fwhm_G/(2*torch.sqrt(2*torch.log(torch.tensor(2, device=self.device_kind)))).to(self.device_kind)
        source_sigma_L = source_fwhm_L/(2*torch.sqrt(2*torch.log(torch.tensor(2, device=self.device_kind)))).to(self.device_kind)

        gauss_filter   = torch.exp(-((sgrid[0]**2 + sgrid[1]**2)*pi_tensor**2*2*source_sigma_G**2)).to(self.device_kind)
        lorentz_filter = torch.exp(-source_sigma_L*pi_tensor*torch.sqrt(sgrid[0]**2 + sgrid[1]**2))

        blur_filter    = gauss_filter*ratio_GtoL + lorentz_filter*(1-ratio_GtoL)

        cbed_blur = torch.fft.ifft2(torch.fft.fft2(cbed, dim=axis) * blur_filter.unsqueeze(-1).unsqueeze(-1), dim=axis).real

        return cbed_blur