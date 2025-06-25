# smatrix_optim/optimization.py
import torch
import numpy as np
from matplotlib import pyplot as plt
import scipy.ndimage as ndimage
from tqdm import tqdm

from torch.cuda.amp import autocast, GradScaler

from smatrix_optim.utility import cbed_resize_rotation, center_of_mass_torch, corr_shift_cbed_torch

def run_optimiser(
        smatrix_torch, 
        smatrix_np, 
        measured_data,
        initial_params, 
        data_loader_params,
        config,
        device
        ):
    """
    Run the optimization loop for S-matrix refinement.
    
    Parameters:
      smatrix_torch : The torch version of the S-matrix object.
      smatrix_np    : The numpy version (used for beam selection and plotting).
      measured_data : Torch tensor containing the measured 4D STEM data.
      Ug_from_OBF   : FFT of the OBF image (used for setting potential).
      initial_params: Dictionary containing initial tensors:
                      - thickness, defocus, source_fwhm, potential filters, etc.
      data_loader_params: Dictionary with parameters from the data loader.
      config        : A configuration dictionary containing:
                      - n_iter_ary, lr_ary, kmax_cutoff_ary, regularize_factor, etc.
      device        : Torch device (cpu or cuda).
    
    Returns:
      results: Dictionary with final optimized parameters and loss history.
    """
    # crop the measured data
    nky_crop_losscalc = int(config['kmax_crop_loss_calc'] * smatrix_np.k0 / data_loader_params['sampling_diff'][0]*2)
    nkx_crop_losscalc = int(config['kmax_crop_loss_calc'] * smatrix_np.k0 / data_loader_params['sampling_diff'][1]*2)
    com_measured_data = center_of_mass_torch(torch.mean(measured_data, dim=(0,1)), device=device)
    print(nky_crop_losscalc, nkx_crop_losscalc)

    # Crop the measured data to the center
    start_y = int(com_measured_data[0].to('cpu').detach().numpy()-nky_crop_losscalc/2) #round(com_measured_data[0].to('cpu').detach().numpy()-nky_crop_losscalc/2) 
    start_x = int(com_measured_data[1].to('cpu').detach().numpy()-nkx_crop_losscalc/2) #round(com_measured_data[1].to('cpu').detach().numpy()-nkx_crop_losscalc/2)
    end_y   = start_y + nky_crop_losscalc #round(com_measured_data[0].to('cpu').detach().numpy()+nky_crop_losscalc/2) #
    end_x   = start_x + nkx_crop_losscalc #round(com_measured_data[1].to('cpu').detach().numpy()+nkx_crop_losscalc/2) #
    measured_data     = measured_data[:,:,start_y:end_y, start_x:end_x]
    print(measured_data.size())
    
    # Unpack initial parameters (you might include more than these)
    Ug_from_OBF = initial_params['potential_Ug'] # torch.tensor
    thickness   = initial_params['thickness']   # torch.tensor, requires_grad set as in config
    defocus     = initial_params['defocus']     # torch.tensor
    source_fwhm = initial_params['source_fwhm'] # torch.tensor

    tilt_rad       = initial_params['tilt_rad']       # torch.tensor, if applicable
    coma           = initial_params['coma']         # torch.tensor, if applicable
    twofold_stig   = initial_params['twofold_stig'] # torch.tensor, if applicable
    threefold_stig = initial_params['threefold_stig'] # torch.tensor, if applicable

    probe_position = initial_params['probe_position'] # torch.tensor, if applicable
    
    # Lists to store loss and parameter evolution
    loss_history = []
    thickness_history = []
    defocus_history = []
    source_size_history = []    
    tilt_history_y = []
    tilt_history_x = []
    coma_history_y = []
    coma_history_x = []
    twofold_stig_history_y = []
    twofold_stig_history_x = []
    threefold_stig_history_y = []
    threefold_stig_history_x = []

    # Common filters we use for the potential from OBF
    filter_els = torch.ones_like(torch.tensor(Ug_from_OBF), dtype=torch.complex64, device=device) #np.ones(np.shape(Ug_from_OBF), dtype=np.complex64)
    filter_abs = torch.ones_like(torch.tensor(Ug_from_OBF), dtype=torch.complex64, device=device) #np.ones(np.shape(Ug_from_OBF), dtype=np.complex64)
    
    # diff space sampling ratio (estimated to measured data) and rotation offset
    pixel_ratio_diff_est_to_meas_y = smatrix_torch.dqy/data_loader_params['sampling_diff'][0]
    pixel_ratio_diff_est_to_meas_x = smatrix_torch.dqx/data_loader_params['sampling_diff'][1]
    rotation_offset_diff           = data_loader_params['rot_offset_deg'] * np.pi / 180.0

    print("Starting optimization loop...")
    # Create a GradScaler for automatic mixed precision
    #scaler = GradScaler()
    
    # Outer loop over different kmax settings (if you want progressive optimization)
    for i_kcutoff, (lr, n_iter, kmax_cutoff_val) in enumerate(zip(
            config['lr_ary'], config['n_iter_ary'], config['kmax_optim_ary'])):
        # Here you might update beams based on new cutoff values.
        # For example: set beams from the current potential
        kmax_cutoff = kmax_cutoff_val * config['kmax_optim']
        kmin_cutoff = np.where((i_kcutoff==0) or (config['kmin_cutoff_adjust']==False), 0, config['kmax_optim_ary'][i_kcutoff-1])
        # Optionally also update kmin_cutoff as needed:
        kmax_cutoff_beam = (1 - np.max([smatrix_np.dqy, smatrix_np.dqx]) / smatrix_np.k0) * kmax_cutoff
        kmin_cutoff_beam = (1 - np.max([smatrix_np.dqy, smatrix_np.dqx]) / smatrix_np.k0) * kmin_cutoff
        # (update beams from image; typically a call like:)
        smatrix_np.set_beams_from_image(
                                        image_fft=Ug_from_OBF, 
                                        amp_cutoff=0e-10,
                                        show_beams=False, 
                                        cutoff_k_frac=kmax_cutoff_beam,
                                        show_nbeams=False
                                        )
        
        # Get the reference potential from OBF on current beams
        Wg_from_OBF = Ug_from_OBF[smatrix_np.beam_indices[0], smatrix_np.beam_indices[1]]
        
        # Create initial potential factors for optimization.
        # For instance, define potential tensors from the filters:
        Wg_els = filter_els[smatrix_np.beam_indices[0, :smatrix_np.n_beam//2], smatrix_np.beam_indices[1, :smatrix_np.n_beam//2]].clone().detach().requires_grad_(True)
        Wg_abs = filter_abs[smatrix_np.beam_indices[0, :smatrix_np.n_beam//2], smatrix_np.beam_indices[1, :smatrix_np.n_beam//2]].clone().detach().requires_grad_(True)
        
        # Define learning rates for these groups.
        current_lr = lr  # From the config for this stage
        
        # Collect parameter groups for the optimizer
        param_groups = []
        if config['optim_potential']:
            param_groups.append({'params': [Wg_els], 'lr': current_lr})
            param_groups.append({'params': [Wg_abs], 'lr': current_lr})# * config['abs_pot_factor']})
        if config['optim_thickness']:
            param_groups.append({'params': [thickness],
                                 'lr': current_lr * config['lr_factor_thickness'] * thickness.item()})
        if config['optim_defocus']:
            param_groups.append({'params': [defocus],
                                 'lr': current_lr * config['lr_factor_defocus'] * abs(thickness.item())})
        if config['optim_sourcesize']:
            param_groups.append({'params': [source_fwhm],
                                 'lr': current_lr * config['lr_factor_sourcesize'] * source_fwhm.item()})
        if config['optim_tilt']:
            param_groups.append({'params': [tilt_rad],
                                 'lr': current_lr * config['lr_factor_tilt']})
        if config['optim_coma']:
            param_groups.append({'params': [coma],
                                 'lr': current_lr * config['lr_factor_coma']})
        if config['optim_twofold_stig']:
            param_groups.append({'params': [twofold_stig],
                                 'lr': current_lr * config['lr_factor_twofold_stig']})
        if config['optim_threefold_stig']:
            param_groups.append({'params': [threefold_stig],
                                 'lr': current_lr * config['lr_factor_threefold_stig']})
        if config['optim_probe_position']:
            param_groups.append({'params': [probe_position],
                                 'lr': current_lr * config['lr_factor_probe_position']})
    
        # tensor for the initial potential
        obf_initial = torch.tensor(Wg_from_OBF, device=device, requires_grad=False, dtype=torch.complex64)

        # Instantiate optimizer, e.g., Adam
        #if config['amsgrad']:
        #    optimizer = torch.optim.Adam(param_groups, amsgrad=config['amsgrad'])
        #else:
        optimizer = torch.optim.Adam(param_groups)
        

        # Inner loop: run n_iter optimization steps
        for iter_idx in range(n_iter):
        #for iter_idx in tqdm(range(n_iter), desc=f"Optimisation loop "):
            optimizer.zero_grad()

            #with autocast():
            
            # Create a potential image from OBF reference and potential parameters.
            # Create a potential "filter" image for each half:
            pot_optim = torch.stack([torch.tensor(Ug_from_OBF, device=device, dtype=torch.complex64) * filter_els, 
                            torch.tensor(Ug_from_OBF, device=device, dtype=torch.complex64) * config["abs_pot_factor"] * filter_abs], dim=0)
            #torch.tensor(np.array([Ug_from_OBF*filter_els, Ug_from_OBF*config["abs_pot_factor"]*filter_abs]),
                        #            device=device,
                        #            dtype=torch.complex64)
            # First half: use Wg_els
            pot_optim[0, smatrix_np.beam_indices[0, :smatrix_np.n_beam//2],
                        smatrix_np.beam_indices[1, :smatrix_np.n_beam//2]] = \
                obf_initial[:smatrix_np.n_beam//2] * Wg_els
            # Second half: use Wg_abs (scaled)
            pot_optim[1, smatrix_np.beam_indices[0, :smatrix_np.n_beam//2],
                        smatrix_np.beam_indices[1, :smatrix_np.n_beam//2]] = \
                obf_initial[:smatrix_np.n_beam//2] * config['abs_pot_factor'] * Wg_abs

            if kmin_cutoff==0:
                pot_optim[0, smatrix_np.beam_indices[0, smatrix_np.n_beam//2+1:smatrix_np.n_beam], 
                        smatrix_np.beam_indices[1, smatrix_np.n_beam//2+1:smatrix_np.n_beam]] = \
                            obf_initial[smatrix_np.n_beam//2+1:smatrix_np.n_beam] * torch.flip((Wg_els).real + 1j*(-Wg_els).imag, dims=(0,))

                pot_optim[1, smatrix_np.beam_indices[0, smatrix_np.n_beam//2+1:smatrix_np.n_beam],
                        smatrix_np.beam_indices[1, smatrix_np.n_beam//2+1:smatrix_np.n_beam]] = \
                            obf_initial[smatrix_np.n_beam//2+1:smatrix_np.n_beam] * torch.flip((Wg_abs).real + 1j*(-Wg_abs).imag, dims=(0,)) * config['abs_pot_factor']
            else:
                pot_optim[0, smatrix_np.beam_indices[0, smatrix_np.n_beam//2:smatrix_np.n_beam],
                        smatrix_np.beam_indices[1, smatrix_np.n_beam//2:smatrix_np.n_beam]] = \
                            obf_initial[smatrix_np.n_beam//2:smatrix_np.n_beam] * torch.flip((Wg_els).real + 1j*(-Wg_els).imag, dims=(0,))
                pot_optim[1, smatrix_np.beam_indices[0, smatrix_np.n_beam//2:smatrix_np.n_beam],
                        smatrix_np.beam_indices[1, smatrix_np.n_beam//2:smatrix_np.n_beam]] = \
                            obf_initial[smatrix_np.n_beam//2:smatrix_np.n_beam] * torch.flip((Wg_abs).real + 1j*(-Wg_abs).imag, dims=(0,)) * config['abs_pot_factor']

            # Compute the A-matrix from the potential. This is a function call
            Amatrix_tmp = smatrix_torch.make_Amatrix_from_pot_vector_ChatGPT([pot_optim[0], pot_optim[1]])
            
            # Compute the S-matrix from the A-matrix via your Pade approximant function.
            Smatrix_tmp = smatrix_torch.calc_Smatrix_from_Amatrix_Pade_ChatGPT(
                Amatrix=Amatrix_tmp,
                thickness=thickness,
                scale_factor=config['matrix_scale_factor']
            )
            
            # Compute the estimated CBED from the S-matrix.
            estimated_data = torch.zeros(measured_data.shape, device=device, dtype=torch.float32)
            estimated_data = smatrix_torch.calc_CBED_from_Smatrix_vector_from_imported_scan_pos_ChatGPT(
                smatrix=Smatrix_tmp,
                scan_ary=probe_position,
                chunk_size=config['chunk_size_CBED_calc'],
                defocus=defocus,
                tilt_rad=tilt_rad,
                tilt_z=thickness,
                coma=coma,
                twofold_stig=twofold_stig,
                threefold_stig=threefold_stig,
                kmax_calc_frac= config['beam_outer_angle'] * 1.25,
                show_progress=False,
            )
            # Optionally apply partial coherence
            estimated_data = torch.fft.ifftshift(estimated_data, dim=(-2,-1))
            estimated_data = smatrix_torch.source_blur_cbed_GandL(
                estimated_data,
                n_scan=(probe_position.shape[0], probe_position.shape[1]),
                source_fwhm_G=source_fwhm,
                source_fwhm_L=source_fwhm,
                ratio_GtoL=1
            )
            if config['lowpass_measured_data']:
                estimated_data = torch.fft.ifft2(torch.fft.fft2(estimated_data, dim=(-2,-1)) *
                                        ((smatrix_torch.qgrid[0]**2 + smatrix_torch.qgrid[1]**2) < (smatrix_torch.k0 * config['kmax_optim'])**2)
                                        ).real

            # resize and rotate the estimated data to match the measured data    
            nky_resize_estimated = round(estimated_data.size()[2]*pixel_ratio_diff_est_to_meas_y.item())
            nkx_resize_estimated = round(estimated_data.size()[3]*pixel_ratio_diff_est_to_meas_x.item())

            #if config["pixel_ratio_diff_est_to_meas"]==[1,1]:
            estimated_data       = cbed_resize_rotation(estimated_data,
                                                        angle_rad=torch.tensor(-rotation_offset_diff, device=device), 
                                                        size=(nky_resize_estimated, nkx_resize_estimated))
            
            # Crop the estimated data to match the measured data
            if iter_idx == 0:
                com_estimated_data = center_of_mass_torch(torch.mean(estimated_data, dim=(0,1)), device=device)
            start_y = int(com_estimated_data[0].to('cpu').detach().numpy()-nky_crop_losscalc/2)
            start_x = int(com_estimated_data[1].to('cpu').detach().numpy()-nkx_crop_losscalc/2)
            end_y   = start_y + nky_crop_losscalc
            end_x   = start_x + nkx_crop_losscalc
            estimated_data = estimated_data[:,:,start_y:end_y, start_x:end_x]
            
            estimated_data, measured_data = corr_shift_cbed_torch(
                cbed_1=estimated_data/torch.sum(estimated_data, dim=(2,3), keepdims=True),
                cbed_2=measured_data/torch.sum(measured_data, dim=(2,3), keepdims=True),
                crop_y=nky_crop_losscalc,
                crop_x=nkx_crop_losscalc,
                device=device,
            )

            # Calculate the loss (e.g., L2 norm between normalized estimated and measured CBED)
            loss = (torch.norm(estimated_data / torch.sum(estimated_data, dim=(2,3), keepdims=True)
                            - measured_data / torch.sum(measured_data, dim=(2,3), keepdims=True))**2 +
                    config['regularize_factor'] * torch.sum(torch.abs(pot_optim)))
            
            # Backpropagation and parameter update
            loss.backward(retain_graph=True)
            optimizer.step()
            # Use mixed precision for faster training
            #scaler.scale(loss).backward()
            #scaler.step(optimizer)
            #scaler.update()
            
            # Log losses and parameters
            loss_history.append(loss.item())
            thickness_history.append(thickness.item())
            defocus_history.append(defocus.cpu().detach().numpy())
            source_size_history.append(source_fwhm.item())
            tilt_history_y.append(tilt_rad.cpu().detach().numpy())
            tilt_history_x.append(tilt_rad.cpu().detach().numpy())
            coma_history_y.append(coma.cpu().detach().numpy())
            coma_history_x.append(coma.cpu().detach().numpy())
            twofold_stig_history_y.append(twofold_stig.cpu().detach().numpy())
            twofold_stig_history_x.append(twofold_stig.cpu().detach().numpy())
            threefold_stig_history_y.append(threefold_stig.cpu().detach().numpy())
            threefold_stig_history_x.append(threefold_stig.cpu().detach().numpy())

            progress_percent = (iter_idx + 1) / n_iter * 100
            print(f"Iteration {iter_idx + 1}/{n_iter} of kcutoff group {i_kcutoff+1}/{len(config['kmax_optim_ary'])}: Loss = {loss.item():.4f} - {progress_percent:.1f}% complete", end="\r", flush=True)
            
            # Optionally, every N iterations, save intermediate plots or results.
            # (Call your I/O functions here if needed.)
        
        # After finishing the inner loop, update the potentials
        potential_els = pot_optim[0].detach() #.cpu().detach().numpy()
        potential_abs = pot_optim[1].detach() #.cpu().detach().numpy()

        if kmin_cutoff==0:
            potential_els[smatrix_np.beam_indices[0, 0:smatrix_np.n_beam//2], 
                          smatrix_np.beam_indices[1, 0:smatrix_np.n_beam//2]] \
                         = obf_initial[0:smatrix_np.n_beam//2]*Wg_els
            potential_els[smatrix_np.beam_indices[0, smatrix_np.n_beam//2+1:smatrix_np.n_beam], 
                          smatrix_np.beam_indices[1, smatrix_np.n_beam//2+1:smatrix_np.n_beam]] \
                         = obf_initial[smatrix_np.n_beam//2+1:smatrix_np.n_beam] * torch.flip(Wg_els.conj(), dims=(0,))
            potential_abs[smatrix_np.beam_indices[0, 0:smatrix_np.n_beam//2], 
                          smatrix_np.beam_indices[1, 0:smatrix_np.n_beam//2]] \
                         = obf_initial[0:smatrix_np.n_beam//2]*Wg_abs
            potential_abs[smatrix_np.beam_indices[0, smatrix_np.n_beam//2+1:smatrix_np.n_beam], 
                          smatrix_np.beam_indices[1, smatrix_np.n_beam//2+1:smatrix_np.n_beam]] \
                         = obf_initial[smatrix_np.n_beam//2+1:smatrix_np.n_beam] * torch.flip(Wg_abs.conj(), dims=(0,))
        else:
            potential_els[smatrix_np.beam_indices[0, 0:smatrix_np.n_beam//2], 
                          smatrix_np.beam_indices[1, 0:smatrix_np.n_beam//2]] \
                         = obf_initial[0:smatrix_np.n_beam//2]*Wg_els
            potential_els[smatrix_np.beam_indices[0, smatrix_np.n_beam//2:smatrix_np.n_beam], 
                          smatrix_np.beam_indices[1, smatrix_np.n_beam//2:smatrix_np.n_beam]]   \
                         = obf_initial[smatrix_np.n_beam//2:smatrix_np.n_beam] * torch.flip(Wg_els.conj(), dims=(0,))
            potential_abs[smatrix_np.beam_indices[0, 0:smatrix_np.n_beam//2], 
                          smatrix_np.beam_indices[1, 0:smatrix_np.n_beam//2]] \
                         = obf_initial[0:smatrix_np.n_beam//2]*Wg_abs
            potential_abs[smatrix_np.beam_indices[0, smatrix_np.n_beam//2:smatrix_np.n_beam], 
                          smatrix_np.beam_indices[1, smatrix_np.n_beam//2:smatrix_np.n_beam]] \
                         = obf_initial[smatrix_np.n_beam//2:smatrix_np.n_beam] * torch.flip(Wg_abs.conj(), dims=(0,))
        
        # After finishing the inner loop, update the filters
        if kmin_cutoff==0:
            filter_els[smatrix_np.beam_indices[0, 0:smatrix_np.n_beam//2],
                        smatrix_np.beam_indices[1, 0:smatrix_np.n_beam//2]] \
                        = Wg_els
            filter_els[smatrix_np.beam_indices[0, smatrix_np.n_beam//2+1:smatrix_np.n_beam],
                        smatrix_np.beam_indices[1, smatrix_np.n_beam//2+1:smatrix_np.n_beam]] \
                        = torch.flip(Wg_els.conj(), dims=(0,))
            filter_abs[smatrix_np.beam_indices[0, 0:smatrix_np.n_beam//2],
                        smatrix_np.beam_indices[1, 0:smatrix_np.n_beam//2]] \
                        = Wg_abs
            filter_abs[smatrix_np.beam_indices[0, smatrix_np.n_beam//2+1:smatrix_np.n_beam],
                        smatrix_np.beam_indices[1, smatrix_np.n_beam//2+1:smatrix_np.n_beam]] \
                        = torch.flip(Wg_abs.conj(), dims=(0,))
        else:
            filter_els[smatrix_np.beam_indices[0, 0:smatrix_np.n_beam//2],
                        smatrix_np.beam_indices[1, 0:smatrix_np.n_beam//2]] \
                        = Wg_els
            filter_els[smatrix_np.beam_indices[0, smatrix_np.n_beam//2:smatrix_np.n_beam],
                        smatrix_np.beam_indices[1, smatrix_np.n_beam//2:smatrix_np.n_beam]] \
                        = torch.flip(Wg_els.conj(), dims=(0,))
            filter_abs[smatrix_np.beam_indices[0, 0:smatrix_np.n_beam//2],
                        smatrix_np.beam_indices[1, 0:smatrix_np.n_beam//2]] \
                        = Wg_abs
            filter_abs[smatrix_np.beam_indices[0, smatrix_np.n_beam//2:smatrix_np.n_beam],
                        smatrix_np.beam_indices[1, smatrix_np.n_beam//2:smatrix_np.n_beam]] \
                        = torch.flip(Wg_abs.conj(), dims=(0,))



    # Once finished, package the results:
    results = {
        "loss_history": loss_history,
        "thickness_history": thickness_history,
        "defocus_history": defocus_history,
        "source_size_history": source_size_history,
        "tilt_history_y": tilt_history_y,
        "tilt_history_x": tilt_history_x,
        "coma_history_y": coma_history_y,
        "coma_history_x": coma_history_x,
        "twofold_stig_history_y": twofold_stig_history_y,
        "twofold_stig_history_x": twofold_stig_history_x,
        "threefold_stig_history_y": threefold_stig_history_y,
        "threefold_stig_history_x": threefold_stig_history_x,
        "optimised_probe_position": probe_position.cpu().detach().numpy(),

        "potential_els": potential_els.cpu().detach().numpy(),
        "potential_abs": potential_abs.cpu().detach().numpy(),
        "obf_initial": Ug_from_OBF,

        "estimated_cbed": estimated_data.cpu().detach().numpy(),
        "measured_cbed": measured_data.cpu().detach().numpy(),
    }
    print("Iteration complete")
    return results
