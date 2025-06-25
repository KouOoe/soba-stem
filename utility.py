import numpy as np
import torch
import torch.nn.functional as F

###################################
# Constants
###################################

hc    = 12.398                # keV*Å
m0c2  = 511.                  # keV
e     = 1.6022e-19            # C
m0    = 9.10938356e-31        #kg
plank = 6.62607004e-34        # m^2kg/s
kmesh_ratio = 5               # ratio between calculated PCTF k-mesh vs. DoSTEM k-mesh
a0    = 0.5292 # Bohr radii in Å


def wavelength(voltage_kV):
    return hc / np.sqrt( voltage_kV * ( 2 * m0c2 + voltage_kV ) )


# interpolation of images
def interpolate(A,newy,newx):
    A_interpolated = np.zeros((newy,newx)).astype(np.complex128)
    oldy,oldx      = np.shape(A)
    A_q            = np.fft.fft2(A)

    # Over sampling condition
    if oldx < newx and oldy < newy:

        for iy in range(-oldy//2,oldy//2):
            for ix in range(-oldx//2,oldx//2):
                A_interpolated[iy][ix] = A_q[iy][ix]

    # Under sampling condition
    else:
        for iy in range(-newy//2,newy//2):
            for ix in range(-newx//2,newx//2):
                A_interpolated[iy][ix] = A_q[iy][ix]


    A_interpolated = np.fft.ifft2(A_interpolated).real * (float(newx) / float(oldx)) * (float(newy) / float(oldy))
    return A_interpolated


def cbed_resize_rotation(cbed, angle_rad, size, interpolate_mode='bicubic'):
    theta = torch.tensor([
        [torch.cos(angle_rad), -torch.sin(angle_rad), 0],
        [torch.sin(angle_rad), torch.cos(angle_rad), 0]
    ], dtype=cbed.dtype, device=cbed.device).unsqueeze(0)

    theta_repeat = theta.repeat(cbed.size(0), 1, 1)

    grid = F.affine_grid(theta_repeat, cbed.size(), align_corners=False)
    rotated_image = F.grid_sample(cbed, grid, mode=interpolate_mode, padding_mode='zeros', align_corners=False)
    resized_image = F.interpolate(rotated_image, size=size, mode=interpolate_mode) 
    
    return resized_image


def center_of_mass_torch(input_tensor, device='cpu'):
    # 入力テンソルの形状を取得
    n, m = input_tensor.shape
    
    # インデックス行列を作成
    x_coords = torch.arange(n, dtype=torch.float32, device=device).view(n, 1).expand(n, m)
    y_coords = torch.arange(m, dtype=torch.float32, device=device).view(1, m).expand(n, m)
    
    # 重み付き座標の合計を計算
    mass = input_tensor.sum()
    x_center = (input_tensor * x_coords).sum() / mass
    y_center = (input_tensor * y_coords).sum() / mass
    
    return y_center, x_center

def subpixel_shift(tensor, shift_y, shift_x):
    """
    Shift a 4D tensor (shape: [n_ry, n_rx, H, W]) in the last two dimensions
    by shift_y and shift_x (in pixel units, can be fractional) using grid_sample.
    
    This function preserves the total intensity by computing a per-image scaling factor.
    
    Parameters:
        tensor : torch.Tensor of shape (n_ry, n_rx, H, W)
        shift_y, shift_x : float (or torch scalar)
    
    Returns:
        shifted : torch.Tensor with the same shape.
    """
    n_ry, n_rx, H, W = tensor.shape
    B = n_ry * n_rx
    tensor_reshaped = tensor.reshape(B, 1, H, W)
    
    # Create normalized grid coordinates for each image.
    xs = torch.linspace(-1, 1, W, device=tensor.device)
    ys = torch.linspace(-1, 1, H, device=tensor.device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')  # (H, W)
    grid = torch.stack((grid_x, grid_y), dim=-1)  # (H, W, 2)
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)    # (B, H, W, 2)
    
    # Convert shift in pixels to normalized shift.
    shift_y_norm = 2 * shift_y / (H - 1)
    shift_x_norm = 2 * shift_x / (W - 1)
    
    # Adjust the grid.
    grid[..., 0] += shift_x_norm
    grid[..., 1] += shift_y_norm
    
    # Apply grid_sample for subpixel interpolation.
    shifted = F.grid_sample(tensor_reshaped, grid, mode='bilinear', padding_mode='border', align_corners=True)
    
    # Compute scaling factor per image to preserve total intensity.
    orig_sum = tensor_reshaped.sum(dim=[2, 3], keepdim=True)
    shifted_sum = shifted.sum(dim=[2, 3], keepdim=True)
    scale = orig_sum / (shifted_sum + 1e-8)
    shifted = shifted * scale
    
    shifted = shifted.reshape(n_ry, n_rx, H, W)
    return shifted

def corr_shift_cbed_torch(cbed_1, cbed_2, crop_y, crop_x, device='cpu'):
    """
    Align two 4D CBED datasets (cbed_1 and cbed_2) so that their PACBED patterns match.
    This function:
      1. Computes the PACBED pattern (averaging over probe positions) for each dataset.
      2. Computes the center-of-mass (COM) of each PACBED.
      3. Calculates the shift needed (subpixel) as the difference between the COMs.
      4. Shifts cbed_1 by that displacement using subpixel interpolation (with intensity preservation).
      5. Crops both datasets in the diffraction dimensions (last two axes) using the COM of cbed_2 as reference.
    
    Parameters:
        cbed_1, cbed_2 : torch.Tensor
            4D tensors of shape (n_ry, n_rx, H, W)
        crop_y, crop_x : int
            Desired crop dimensions (in pixels) in the diffraction (ky, kx) domain.
    
    Returns:
        aligned_cbed_1, aligned_cbed_2 : torch.Tensor
            The aligned and cropped CBED datasets.
    """
    # Compute PACBED patterns by averaging over the probe positions.
    pacbed_1 = cbed_1.mean(dim=(0, 1))
    pacbed_2 = cbed_2.mean(dim=(0, 1))
    
    # Compute COM for each PACBED.
    com1 = center_of_mass_torch(pacbed_1, device=device)
    com2 = center_of_mass_torch(pacbed_2, device=device)
    
    # Compute the shift needed: shift = com2 - com1
    shift_y = com2[0] - com1[0]
    shift_x = com2[1] - com1[1]
    
    # Shift cbed_1 using the subpixel shift function.
    aligned_cbed_1 = subpixel_shift(cbed_1, -shift_y, -shift_x)
    
    # For cropping, we use the COM of cbed_2 as the reference center.
    cy2 = (com2[0].item())
    cx2 = (com2[1].item())
    y0 = max(0, int(cy2 - crop_y / 2))
    x0 = max(0, int(cx2 - crop_x / 2))
    y1 = y0 + crop_y
    x1 = x0 + crop_x
    
    # Crop the diffraction domain (last two dimensions) of both datasets.
    aligned_cbed_1_crop = aligned_cbed_1[:, :, y0:y1, x0:x1]
    cbed_2_crop = cbed_2[:, :, y0:y1, x0:x1]
    
    return aligned_cbed_1_crop, cbed_2_crop


def add_noise_to_cbed(cbed, electron_dose):
    return np.random.poisson(cbed * electron_dose) / electron_dose