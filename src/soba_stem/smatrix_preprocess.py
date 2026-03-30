import numpy as np
import scipy

from matplotlib import pyplot as plt
import sys, math


# constant
hc   = 12.398                # keV*Å
m0c2 = 511.                  # keV
e    = 1.6022e-19            # C



class SMatrixPreProcess():

    def __init__(self, voltage, aperture_mrad):
        super().__init__()

        self.voltage    = voltage
        self.aperture   = aperture_mrad

        self.wavelength = hc / np.sqrt( self.voltage * ( 2 * m0c2 + self.voltage ) ) # Å
        self.k0         = self.aperture * 1e-3 / self.wavelength                     # Å-1
        self.zprobe     = 1 / self.k0**2 / self.wavelength


    def make_grid(self, ry, rx, ny, nx):
        self.ry  = ry
        self.rx  = rx
        self.nry = ny
        self.nrx = nx
        self.dry = ry/ny
        self.drx = rx/nx
        self.dqy = 1/ry
        self.dqx = 1/rx

        self.rgrid = np.meshgrid(np.arange(0, ry, self.dry), np.arange(0, rx, self.drx), indexing='ij')
        self.qgrid = np.meshgrid(np.fft.fftfreq(self.nry, self.dry), np.fft.fftfreq(self.nrx, self.drx), indexing='ij')


    def set_beams_from_image(self, image_fft, amp_cutoff=0, show_beams=True, cutoff_k_frac=1, cutoff_k_inner_frac=0, show_nbeams=True):
        
        ny_crop       = int(self.k0/self.dqy * 2)
        nx_crop       = int(self.k0/self.dqx * 2)

        pot_fft_crop = np.fft.fftshift(image_fft)[self.nry//2-ny_crop:self.nry//2+ny_crop,self.nrx//2-nx_crop:self.nrx//2+nx_crop]
        cutoff_k     = (np.sqrt(self.qgrid[0]**2 + self.qgrid[1]**2) < self.k0 * cutoff_k_frac) & (np.sqrt(self.qgrid[0]**2 + self.qgrid[1]**2) >= self.k0 * cutoff_k_inner_frac)
        beam_area    = cutoff_k & (np.abs(image_fft)>=amp_cutoff)

        if cutoff_k_inner_frac==0:
            beam_area[0,0] = True # to ensure to include the 00 beam.
        else:
            beam_area[0,0] = False
        self.n_beam         = np.sum(beam_area)


        sorted_coordinates, sorted_indices = self.sorted_coordinates_within_cutoff(image_fft, cutoff_k, amp_cutoff=amp_cutoff)

        if cutoff_k_inner_frac==0:
            self.beam_coordinates = np.concatenate((-np.flip(sorted_coordinates[0]), np.array([0]), sorted_coordinates[0])), np.concatenate((-np.flip(sorted_coordinates[1]), np.array([0]), sorted_coordinates[1]))
            self.beam_indices     = np.array([(self.beam_coordinates[0]/self.dqy), (self.beam_coordinates[1]/self.dqx)]).round().astype(int)
        else:
            self.beam_coordinates = np.concatenate((-np.flip(sorted_coordinates[0]), sorted_coordinates[0])), np.concatenate((-np.flip(sorted_coordinates[1]), sorted_coordinates[1]))
            self.beam_indices     = np.array([(self.beam_coordinates[0]/self.dqy), (self.beam_coordinates[1]/self.dqx)]).round().astype(int)

        if show_beams:
            plt.figure(figsize=(10,6))
            plt.subplot(2,3,1)
            plt.imshow((np.abs(pot_fft_crop))**0.5)
            plt.title("Input coodinate")
            plt.subplot(2,3,2)
            plt.imshow(np.fft.fftshift(beam_area)[self.nry//2-ny_crop:self.nry//2+ny_crop,self.nrx//2-nx_crop:self.nrx//2+nx_crop])
            plt.title("Scattering vector")
            plt.subplot(2,3,3)
            plt.imshow(np.fft.fftshift(cutoff_k)[self.nry//2-ny_crop:self.nry//2+ny_crop,self.nrx//2-nx_crop:self.nrx//2+nx_crop])
            plt.title("Cutoff region")
            plt.subplot(2,3,4)
            plt.scatter(self.beam_coordinates[1], self.beam_coordinates[0], c=self.beam_coordinates[0]**2+self.beam_coordinates[1]**2)
            plt.title("Beam coordinates")
            plt.subplot(2,3,5)
            plt.scatter(self.beam_indices[1], self.beam_indices[0], c=self.beam_indices[0]**2+self.beam_indices[1]**2)
            plt.title("Beam indices")
            plt.subplot(2,3,6)
            plt.scatter(self.beam_coordinates[1], self.beam_coordinates[0], c=np.arange(self.n_beam), cmap='bwr')
            plt.title("Beam ordering")
        
        if show_nbeams:
            print("number of beams: ", self.n_beam)


    def sorted_coordinates_within_cutoff(self, image_fft, cutoff_k, amp_cutoff=0):
        # Step 1: Calculate distance from the origin for each coordinate
        distances = np.sqrt(self.qgrid[0]**2 + self.qgrid[1]**2)
        
        # Step 2: Filter coordinates within the cutoff
        y_indices, x_indices = np.where((cutoff_k) & (np.arctan2(self.qgrid[0], self.qgrid[1]) > 0) & (np.abs(image_fft)>=amp_cutoff))
        filtered_distances   = distances[(cutoff_k) & (np.arctan2(self.qgrid[0], self.qgrid[1]) > 0) & (np.abs(image_fft)>=amp_cutoff)]
        filtered_ky = self.qgrid[0][(cutoff_k) & (np.arctan2(self.qgrid[0], self.qgrid[1]) > 0) & (np.abs(image_fft)>=amp_cutoff)]
        filtered_kx = self.qgrid[1][(cutoff_k) & (np.arctan2(self.qgrid[0], self.qgrid[1]) > 0) & (np.abs(image_fft)>=amp_cutoff)]
        
        # Step 3: Sort by distance
        sort_indices = np.argsort(filtered_distances)
        
        # Step 4: For coordinates with the same distance, sort them in clockwise order
        unique_distances, unique_starts, unique_counts = np.unique(filtered_distances[sort_indices], return_index=True, return_counts=True)
        for start, count in zip(unique_starts, unique_counts):
            if count > 1:
                subset = sort_indices[start:start+count]
                angles = np.arctan2(filtered_ky[subset], filtered_kx[subset])
                subset_sorted = subset[np.argsort(angles)]
                sort_indices[start:start+count] = subset_sorted
        
        # Return the sorted coordinates and their original indices in the 2D array
        sorted_ky = filtered_ky[sort_indices]
        sorted_kx = filtered_kx[sort_indices]
        sorted_y_indices = y_indices[sort_indices]
        sorted_x_indices = x_indices[sort_indices]
        
        return (sorted_ky, sorted_kx), (sorted_y_indices, sorted_x_indices)
    

    def Ug_circle_plot(self, Ug_1, Ug_2, display_factor_kmax=2.5, circle_size_factor=150, size_cutoff=0.1):

        kmax_display = self.aperture*1e-3*display_factor_kmax/self.wavelength
        size_factor  = circle_size_factor/np.max(np.abs(Ug_2))

        Ug_1[0,0] = 0
        Ug_2[0,0] = 0

        #plt.figure(figsize=(5,4))

        plt.scatter(self.qgrid[0], self.qgrid[1], s=np.where(np.abs(Ug_1)*size_factor>size_cutoff, np.abs(Ug_1)*size_factor, 0), c=np.mod(np.angle(Ug_1)-np.angle(Ug_2)+np.pi, np.pi*2)-np.pi, cmap='hsv', vmin=-np.pi, vmax=np.pi, alpha=0.5)
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=14)
        plt.scatter(self.qgrid[0], self.qgrid[1], s=np.where(np.abs(Ug_2)*size_factor>size_cutoff, np.abs(Ug_2)*size_factor, 0), alpha=1, linewidths=1, edgecolors='black', c='None')

        #plot a circle indicating an aperture size
        theta = np.arange(0,2*np.pi,0.01)
        plt.plot(self.aperture*1e-3/self.wavelength*np.cos(theta),self.aperture*1e-3/self.wavelength*np.sin(theta), linestyle='--', color='grey', linewidth=1.44)


        plt.xlim(-kmax_display,kmax_display)
        plt.ylim(-kmax_display,kmax_display)

        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        plt.ylabel('Spatial frequency ($\mathrm{\AA}^{-1}$)', fontsize=16)
        plt.xlabel('Spatial frequency ($\mathrm{\AA}^{-1}$)', fontsize=16)

        #plt.show()