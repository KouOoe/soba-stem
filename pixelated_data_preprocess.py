import numpy as np
from tifffile import imsave, imread
from scipy import ndimage
import sys

### constant
hc    = 12.398      # keV*Å
m0c2  = 511.        # keV


class PixelatedDataProcess():

    def __init__(self, voltage, aperture):

        self.voltage = voltage
        self.aperture = aperture

        self.wavelength = hc / np.sqrt( self.voltage * ( 2 * m0c2 + self.voltage ) ) # Å
        self.k0         = self.aperture * 1e-3 / self.wavelength                     # Å-1
        self.zprobe     = 1 / self.k0**2 / self.wavelength

    def A(self,ky,kx):
        aperture = (ky**2 + kx**2 <= self.k0**2)/np.sqrt(np.pi*self.k0**2)
        return aperture
    
    def D(self,ky,kx,inner,outer,theta1=-np.pi,theta2=np.pi,rot_offset=0):
        # inner/outer angle is normalized by aperture angle i.e. r=1 means edge of BF disc.
        inner = inner * self.k0
        outer = outer * self.k0
        rot_epsilon = 1e-8

        kx_rot = kx*np.cos(rot_offset+rot_epsilon) + ky*np.sin(rot_offset+rot_epsilon)
        ky_rot =-kx*np.sin(rot_offset+rot_epsilon) + ky*np.cos(rot_offset+rot_epsilon)

        mask_radius   = np.logical_and(ky**2 + kx**2 < outer**2, ky**2 + kx**2 >= inner**2)
        mask_azimuth  = np.logical_and(np.arctan2(ky_rot, kx_rot) >= theta1, np.arctan2(ky_rot, kx_rot) < theta2)
        mask          = np.logical_and(mask_radius, mask_azimuth)
        
        return mask


    def chi(self,ky,kx,defocus=0,tilt_rad=(0,0)):
        chi = np.pi * defocus * (self.wavelength * (ky**2 + kx**2) - 2*ky*np.tan(tilt_rad[0]) - 2*kx*np.tan(tilt_rad[1]))
        return chi

    def T(self,ky,kx,defocus=0,tilt_rad=(0,0)):
        lens_transfer = self.A(ky,kx) * np.exp(-1j*self.chi(ky,kx,defocus,tilt_rad))
        return lens_transfer


    def tau_calc(self,thickness):
        tau = thickness/self.zprobe
        return tau


    def load_parameter(self, thickness, defocus, dry, drx, dky, dkx, rot_offset_rad=0):

        self.thickness = thickness
        self.defocus   = defocus
        self.rot_offset = rot_offset_rad #0 # rad, not deg.
        self.dry  = dry
        self.drx  = drx
        self.dky  = dky
        self.dkx  = dkx



    def cbed_import(self, cbed_data):

        self.cbed_data = cbed_data
        self.pacbed = np.sum(cbed_data, axis=(0,1))
        self.centroid = ndimage.measurements.center_of_mass(self.pacbed)
        
        self.nry, self.nrx, self.nky, self.nkx = np.shape(cbed_data)

        self.kgrid = np.meshgrid(
                        np.arange(-self.centroid[0],(np.shape(self.pacbed)[0]-self.centroid[0]),1)*self.dky,
                        np.arange(-self.centroid[1],(np.shape(self.pacbed)[1]-self.centroid[1]),1)*self.dkx,
                        indexing='ij'
                        )

        self.ky, self.kx = self.nky*self.dky, self.nkx*self.dkx
        self.ry, self.rx = self.nry*self.dry, self.nrx*self.drx

        self.qgrid = np.meshgrid(
            np.fft.fftfreq(self.nry, d=self.dry),
            np.fft.fftfreq(self.nrx, d=self.drx),
            indexing='ij'
        )

        print('4D data imported!')
    



    def OBF_pixel_mask(self, qy, qx, thickness_input=None):

        if thickness_input==None:
            thickness_input = self.thickness
        else:
            pass

        zmesh   = self.zprobe/4
        n_slice = int(thickness_input/zmesh)

        if n_slice>5:
            pass
        else:
            n_slice = 5
            zmesh   = thickness_input/n_slice

        kx_rot =  self.kgrid[1]*np.cos(self.rot_offset) + self.kgrid[0]*np.sin(self.rot_offset)
        ky_rot = -self.kgrid[1]*np.sin(self.rot_offset) + self.kgrid[0]*np.cos(self.rot_offset)

        mask = np.zeros((self.nky,self.nkx), dtype='complex64')

        for islice in range(n_slice):
            z     = self.defocus + zmesh*islice
            mask += np.conj(self.T(ky_rot, kx_rot, z)) * self.T(ky_rot-qy, kx_rot-qx, z) - self.T(ky_rot, kx_rot, z) * np.conj(self.T(ky_rot+qy, kx_rot+qx, z))
        
        mask = np.conj(mask)/n_slice

        return mask



    def OBF_pixel(self, sourcesize=0, ratio_GtoL=0.5, normalize_mode='noise', thickness_input=None, output_mode='real'):

        kx_rot =  self.kgrid[1]*np.cos(self.rot_offset) + self.kgrid[0]*np.sin(self.rot_offset)
        ky_rot = -self.kgrid[1]*np.sin(self.rot_offset) + self.kgrid[0]*np.cos(self.rot_offset)

        cbed         = self.cbed_data
        intensity_BF = np.sum(np.sum(cbed*self.D(ky_rot,kx_rot,0,1,-np.pi,np.pi),axis=3),axis=2)
        cbed         = cbed / intensity_BF[:,:,np.newaxis,np.newaxis]

        pctf        = np.zeros((self.nry,self.nrx), dtype='float32')
        phase       = np.zeros((self.nry,self.nrx), dtype='complex64')

        g_func = np.fft.fft2(self.cbed_data, axes=(0,1))*self.ry*self.rx/(self.nry*self.nrx)

        if sourcesize>0:
            sourcesize_filter = self.source_blur_filter(source_fwhm=sourcesize, ratio_GtoL=ratio_GtoL)

        for iy in range(-self.nry//2, self.nry//2):
            for ix in range(-self.nrx//2, self.nrx//2):
                qy, qx = 1/self.ry*iy, 1/self.rx*ix

                #mask = self.OBF_pixel_mask(qy,qx)
                #pctf[iy,ix] = (np.sum(mask*np.conj(mask)/(np.pi*self.k0**2))).real*self.dky*self.dkx

                '''
                if qy**2+qx**2>0:
                    phase[iy,ix] = np.sum(g_func[iy,ix]*mask/(np.pi*self.k0**2))*self.dky*self.dkx
                else:
                    phase[iy,ix] = 0
                '''
                if qy**2+qx**2>0 and qy**2+qx**2<=(2*self.k0)**2:
                    mask = self.OBF_pixel_mask(qy,qx,thickness_input)

                    if sourcesize>0:
                        mask *= sourcesize_filter[iy,ix]

                    pctf[iy,ix]  = (np.sum(mask*np.conj(mask)/(np.pi*self.k0**2))).real*self.dky*self.dkx
                    phase[iy,ix] = np.sum(g_func[iy,ix]*mask/(np.pi*self.k0**2))*self.dky*self.dkx

                else:
                    pctf[iy,ix]  = 0
                    phase[iy,ix] = 0

                progress = ((iy+self.nry//2)*self.nrx+(ix+1)+self.nrx//2)/(self.nry*self.nrx)*100
                sys.stdout.write('\r{:.1f}% done...'.format(progress))
                sys.stdout.flush()

        if output_mode=='real':

            if normalize_mode=='noise':
                noise = np.sqrt(pctf)
                scale = np.max(abs(noise))/np.max(abs(pctf))
                obf   = phase * np.conj(noise)/(np.conj(noise)*noise+np.max(np.abs(noise))*1e-2) * scale
                obf   = np.fft.ifft2(obf).imag/self.ry/self.rx*self.nry*self.nrx
            elif normalize_mode=='ctf':
                obf   = phase * np.conj(pctf)/(np.conj(pctf)*pctf+np.max(np.abs(pctf))*1e-2)
                obf   = np.fft.ifft2(obf).imag/self.ry/self.rx*self.nry*self.nrx

        elif output_mode=='complex':

            if normalize_mode=='noise':
                noise = np.sqrt(pctf)
                scale = np.max(abs(noise))/np.max(abs(pctf))
                obf   = phase * np.conj(noise)/(np.conj(noise)*noise+np.max(np.abs(noise))*1e-2) * scale
                obf   = np.fft.ifft2(obf)/self.ry/self.rx*self.nry*self.nrx
            elif normalize_mode=='ctf':
                obf   = phase * np.conj(pctf)/(np.conj(pctf)*pctf+np.max(np.abs(pctf))*1e-2)
                obf   = np.fft.ifft2(obf)/self.ry/self.rx*self.nry*self.nrx

        return obf/self.dky/self.dkx