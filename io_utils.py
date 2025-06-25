import matplotlib.pyplot as plt
from tifffile import imwrite
import numpy as np
import torch

def save_image(image, filename):
    """
    Save a 2D image to a TIFF file.
    
    Parameters:
        image (numpy.ndarray): The image to save.
        filename (str): The path to the output TIFF file.
    """
    # Ensure the image is in the correct format
    if len(image.shape) == 2:
        imwrite(filename, image.astype(np.float32), photometric='minisblack')
    else:
        raise ValueError("Image must be a 2D array.")
    
    plt.figure(figsize=(4,4))
    plt.imshow(image)
    plt.axis('off')
    plt.savefig(filename.replace('.tiff', '.png'), bbox_inches='tight')
    plt.close()
    

def save_scan_position_diff(scan_position_init, scan_position_optim, save_dir, figsize=(8, 6)):
    """
    Save scan positions to a text file.
    
    Parameters:
        scan_positions (numpy.ndarray): The scan positions to save.
        filename (str): The path to the output text file.
    """
    diff_scan_pos = scan_position_optim - scan_position_init
    scan_position_init_y = scan_position_init[:,:,0].flatten()
    scan_position_init_x = scan_position_init[:,:,1].flatten()
    diff_scan_pos_y = diff_scan_pos[:,:,0].flatten()
    diff_scan_pos_x = diff_scan_pos[:,:,1].flatten()


    plt.figure(figsize=figsize)

    # 座標 (pos_x_flat, pos_y_flat) の点を始点とし、そこから (dif_x_flat, dif_y_flat) のベクトルを描画
    plt.quiver(
        scan_position_init_x,      # 各点の X 座標
        scan_position_init_y,      # 各点の Y 座標
        diff_scan_pos_x,      # X方向の変位（差分）
        diff_scan_pos_y,      # Y方向の変位（差分）
        angles='xy', scale_units='xy', scale=1.0, color='red'
    )

    plt.title("Optimised - Initial (Vector Field)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis('equal')  # アスペクト比を等しくする
    plt.grid(True)

    plt.savefig(save_dir + "/Scan_potion_optim_vector.png", bbox_inches='tight')
    plt.close()


def save_cbed_diff(cbed_1, cbed_2, save_dir, probe_pos=(0,0), figsize=(12,3)):
    """
    Save the difference between two CBED patterns.
    
    Parameters:
        cbed_1 (numpy.ndarray): The first CBED pattern.
        cbed_2 (numpy.ndarray): The second CBED pattern.
        filename (str): The path to the output text file.
    """
    cbed_1 = cbed_1/np.sum(cbed_1, axis=(2,3), keepdims=True)
    cbed_2 = cbed_2/np.sum(cbed_2, axis=(2,3), keepdims=True)

    plt.figure(figsize=figsize)
    plt.subplot(131)
    plt.imshow(cbed_1[probe_pos[0], probe_pos[1]])
    plt.title("Estimated")
    plt.colorbar()
    plt.subplot(132)
    plt.imshow(cbed_2[probe_pos[0], probe_pos[1]])
    plt.title("Measured")
    plt.colorbar()
    plt.subplot(133)
    plt.imshow(cbed_1[probe_pos[0], probe_pos[1]] - cbed_2[probe_pos[0], probe_pos[1]])
    plt.colorbar()
    plt.title('Diff.')

    plt.savefig(save_dir + "/CBED_comparison_{:}_{:}.png".format(probe_pos[0], probe_pos[1]), bbox_inches='tight')
    plt.close()

    plt.figure(figsize=figsize)
    plt.subplot(131)
    plt.imshow(np.sum(cbed_1, axis=(0,1)))
    plt.title("Estimated")
    plt.colorbar()
    plt.subplot(132)
    plt.imshow(np.sum(cbed_2, axis=(0,1)))
    plt.title("Measured")
    plt.colorbar()
    plt.subplot(133)
    plt.imshow(np.sum(cbed_1, axis=(0,1)) - np.sum(cbed_2, axis=(0,1)))
    plt.colorbar()
    plt.title('Diff.')
    plt.savefig(save_dir + "/PACBED_comparison.png", bbox_inches='tight')
    plt.close()


def plot_params_history_main(optim_results, save_dir, figsize=(3,8)):

    plt.figure(figsize=figsize)
    plt.subplot(4,1,1)
    plt.plot(optim_results["loss_history"])
    plt.ylabel('Loss', fontsize=10)
    plt.subplot(4,1,2)
    plt.plot(optim_results["thickness_history"])
    plt.ylabel('Thickness', fontsize=10)
    plt.subplot(4,1,3)
    plt.plot(optim_results["defocus_history"])
    plt.xlabel('Iteration', fontsize=10)    
    plt.ylabel('Defocus', fontsize=10)
    plt.subplot(4,1,4)
    plt.plot(optim_results["source_size_history"])
    plt.ylabel('Source size', fontsize=10)
    plt.xlabel('Iteration', fontsize=10)
    plt.savefig(save_dir + "/params_history_main.png", bbox_inches='tight')
    plt.close()

def plot_params_history_aberrations(optim_results, save_dir, figsize=(3,8)):

    plt.figure(figsize=figsize)
    plt.subplot(4,1,1)
    plt.plot(optim_results["twofold_stig_history_y"])
    plt.plot(optim_results["twofold_stig_history_x"])
    plt.ylabel('Twofold stig', fontsize=10)
    plt.subplot(4,1,2)
    plt.plot(optim_results["coma_history_y"])
    plt.plot(optim_results["coma_history_x"])
    plt.ylabel('Coma', fontsize=10)
    plt.subplot(4,1,3)
    plt.plot(optim_results["tilt_history_y"])
    plt.plot(optim_results["tilt_history_x"])
    plt.ylabel('Tilt', fontsize=10)
    plt.subplot(4,1,4)
    plt.plot(optim_results["threefold_stig_history_y"])
    plt.plot(optim_results["threefold_stig_history_x"])
    plt.ylabel('Threefold stig', fontsize=10)

    plt.savefig(save_dir + "/params_history_aberrations.png", bbox_inches='tight')
    plt.close()

def potential_circle_plot(smatrix_np, optimised_Ug, reference_Ug, save_dir, figsize=(5,4)):
    
    smatrix_np.Ug_circle_plot(optimised_Ug, 
                              reference_Ug, 
                              display_factor_kmax=2, 
                              circle_size_factor=250, 
                              size_cutoff=0.01)
    
    plt.savefig(save_dir + "/potential_circle_plot.png", bbox_inches='tight')
    plt.close()

def save_optimised_potential_image(
        Ug_optimised,
        Ug_reference,
        Ug_obf,
        save_dir,
        figsize=(12,5),
):
    obf_realspace = np.fft.ifft2(Ug_obf).real
    pot_optimised = np.fft.ifft2(Ug_optimised).real
    pot_reference = np.fft.ifft2(Ug_reference).real

    plt.figure(figsize=figsize)
    plt.subplot(131)
    plt.imshow(obf_realspace)
    plt.title("OBF")
    plt.subplot(132)
    plt.imshow(pot_optimised)
    plt.title("Optimised")
    plt.subplot(133)
    plt.imshow(pot_reference)
    plt.title("Reference")

    plt.savefig(save_dir + "/potential_image.png", bbox_inches='tight')
    plt.close()
