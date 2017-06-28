import numpy as np
import math
from scipy.integrate import cumtrapz
from scipy.integrate import simps
from scipy import interpolate
from scipy.special import jn
from scipy.special import erf


class Blackbody(object):
    """
    This class calculates blackbody radiation using the Planck equation
    SI units used internally, expects wavlength in um
    """

    def __init__(self, T):
        """
        :param1 T: float, Temperature in K
        """
        self.tempK = T
        # Planck constant
        self.h = 6.6262e-34
        # speed of light
        self.c = 2.9979e08
        # Boltzman constant
        self.k = 1.38062e-23
        # c2 constant for Planck radiation law
        self.c2 = self.h * self.c / self.k

    def total_excitance(self):
        """
        Stefan Boltzman law

        :returns: total exitance in watts/m^2 integrated over all wavelengths
        """
        sigma = (2 * math.pow(math.pi, 5) * math.pow(self.k, 4)) / (15 * math.pow(self.h, 3) * math.pow(self.c, 2))
        return sigma * math.pow(self.tempK, 4)

    def peak_wvl(self):
        """
        :returns: peak wavlength in um from Wien displacment law
        """
        lambdam = 2.8978e-03 / self.tempK
        return lambdam * 1e6

    def _power(self, wvls):
        """
        Computes the Planck Radiation law without the c1 term
        P = (wvl)^-5 / ( exp(c2/wvl T) -1  )

        :param1 wvls: 1D array of wavlengths in um

        :returns: 1D numpy array
        """
        wvls_m = wvls * 1e-06
        n = np.size(wvls_m)
        numerator = np.power(wvls_m, -5 * np.ones(n))
        scale = self.c2 / self.tempK
        temp = scale * np.ones(n)
        temp = np.divide(temp, wvls_m)
        denominator = np.expm1(temp)
        return np.divide(numerator, denominator)

    def exitance(self, wvls, polarized=False):
        """
        Computes the unpolarized or polarized spectral exitance in
        watts/m^2 m

        :param1 wvls: 1D array of wavlengths in um
        :param2 polarized: type of exitance to return
        :returns: 1D numpy array
        """
        if not polarized:
            c1 = 2 * math.pi * self.h * math.pow(self.c, 2)
        else:
            c1 = math.pi * self.h * math.pow(self.c, 2)
        return c1 * self._power(wvls)

    def radiance(self, wvls, polarized=False):
        """
        Computes the unpolarized or polarized spectral radiance in
        watts/m^2 m sr

        :param1 wvls: 1D array of wavlengths in um
        :param2 polarized: type of radiance to return
        :returns: 1D numpy array
        """
        if not polarized:
            c1 = 2 * self.h * math.pow(self.c, 2)
        else:
            c1 = self.h * math.pow(self.c, 2)
        return c1 * self._power(wvls)

    def bb_spectral_weights(self, wvls, responsivity, n):
        """
        Computes N wavelengths (in um) for unit amplitude spectral
        weights, based on the total power between wvl_short and wvl_long.
        Each wavelength sits at the center wavelength of a band of power
        containing 1/N the total power between wvl_short and wvl_long.

        :param1 wvls: array of wavlength in um
        :param2 responsivity: spectral responsivity of detector at wvls
        :param3 n: number of spectral weights desired
        :returns: list of wavelengths in um
        """

        p = self._power(wvls)
        weighted_p = np.multiply(p, responsivity)
        cum_p = cumtrapz(weighted_p, wvls, initial=0.0)
        total = cum_p[-1]
        targets = [total / (2 * n) + (i * total / n) for i in range(n)]
        roots = []
        for t in targets:
            y = np.subtract(cum_p, t * np.ones(np.size(cum_p)))
            tck = interpolate.splrep(wvls, y, s=0)
            roots.append(interpolate.sproot(tck)[0])
        return roots

    def luminance(self, polarized=False):
        """
        Calculates luminance in lumens/ m^2 sr

        :returns: luminance
        """
        wvl = np.asarray(
            [0.380, 0.385, 0.390, 0.395, 0.400, 0.405, 0.410, 0.415, 0.420, 0.425, 0.430, 0.435, 0.440, 0.445, 0.450,
             0.455, 0.460, 0.465, 0.470, 0.475, 0.480, 0.485, 0.490, 0.495, 0.500, 0.505, 0.510, 0.515, 0.520, 0.525,
             0.530, 0.535, 0.540, 0.545, 0.550, 0.555, 0.560, 0.565, 0.570, 0.575, 0.580, 0.585, 0.590, 0.595, 0.600,
             0.605, 0.610, 0.615, 0.620, 0.625, 0.630, 0.635, 0.640, 0.645, 0.650, 0.655, 0.660, 0.665, 0.670, 0.675,
             0.680, 0.685, 0.690, 0.695, 0.700, 0.705, 0.710, 0.715, 0.720, 0.725, 0.730])
        vlambda = np.asarray(
            [3.9000E-05, 6.4000E-05, 1.2000E-04, 2.1700E-04, 3.9600E-04, 6.4000E-04, 1.2100E-03, 2.1800E-03, 4.0000E-03,
             7.3000E-03, 1.1600E-02, 1.6840E-02, 2.3000E-02, 2.9800E-02, 3.8000E-02, 4.8000E-02, 6.0000E-02, 7.3900E-02,
             9.0980E-02, 1.1260E-01, 1.3902E-01, 1.6930E-01, 2.0802E-01, 2.5860E-01, 3.2300E-01, 4.0730E-01, 5.0300E-01,
             6.0820E-01, 7.1000E-01, 7.9320E-01, 8.6200E-01, 9.1485E-01, 9.5400E-01, 9.8030E-01, 9.9495E-01, 1.0000E+00,
             9.9500E-01, 9.7860E-01, 9.5200E-01, 9.1540E-01, 8.7000E-01, 8.1630E-01, 7.5700E-01, 6.9490E-01, 6.3100E-01,
             5.6680E-01, 5.0300E-01, 4.4120E-01, 3.8100E-01, 3.2100E-01, 2.6500E-01, 2.1700E-01, 1.7500E-01, 1.3820E-01,
             1.0700E-01, 8.1600E-02, 6.1000E-02, 4.4580E-02, 3.2000E-02, 2.3200E-02, 1.7000E-02, 1.1920E-02, 8.2100E-03,
             5.7230E-03, 4.1020E-03, 2.9290E-03, 2.0910E-03, 1.4840E-03, 1.0470E-03, 7.4000E-04, 5.2000E-04])
        Km = 673.0
        r = self.radiance(wvl, polarized)
        wvl = 1e-06 * wvl
        L = Km * simps(np.multiply(r, vlambda), wvl)
        return L


def fwhm_angle(wvl, omega, aprad):
    """
    :param wvl: wavlength in same units as other quantities
    :param omega: exp(-2) semi-diameter of the spot irradiance profile
    :param aprad: semi-diameter of truncating aperture
    :return: returns an estimate of the FWHM divergence in radians
    """
    x = float(aprad) / float(omega)
    if x > 1.0:
        x = math.pow(x, 2)
        C1 = 1.0 + 2 * x / 7
        C2 = 1.0 + 2 * C1 * x / 5
        C3 = 1.0 + 2 * C2 * x / 3
        C4 = 15 * C2 / (2 * C1 * x)
        z = 1.0 - .195262 * 6 * C1 * C3 / (5 * math.pow(C2, 2))
        bigX = 1.03305 * C4 * (1.0 - math.sqrt(z))
        return wvl * math.sqrt(2 * bigX) / (math.pi * omega)
    else:
        C1 = math.sqrt(math.pi) * erf(x)
        C2 = (C1 / 2) - x * math.exp(-math.pow(x, 2))
        C3 = 1.5 * C2 - math.pow(x, 3) * math.exp(-math.pow(x, 2))
        z = 1.0 - (.195262 * C1 * C3 / math.pow(C2, 2))
        bigX = (3 * C2 * (1 - math.sqrt(z)) / C3)
        return wvl * ((math.sqrt(2 * bigX) / (math.pi * omega)) - .00388 / omega)


def fwhm2omega(fwhm):
    """
    :param fwhm: Full Width Half Maximum value of irradiance profile
    :return: exp(-2) semi-diameter of the irradiance profile
    """
    return fwhm / (2.0 * math.sqrt(-0.5 * math.log(0.5)))


def omega2fwhm(omega):
    """
    :param omega: exp(-2) semi-diameter of the irradiance profile
    :return: Full Width Half Maximum (FWHM) value for the irradiance profile
    """
    return 2.0 * omega * math.sqrt(-0.5 * math.log(0.5))


def airy(wvl, NA, r):
    """
    :param wvl: wavelength in same units as r
    :param NA: Numerical Aperture
    :param r: numpy array of radial distances
    :return: Electric field value of Airy diffraction pattern as numpy array
    """
    x = 2 * math.pi * NA * r / wvl
    numerator = 2 * jn(1,x)
    ret = np.true_divide(numerator, x)
    # replace divide by 0 with 1.0
    ret[np.isnan(ret)] = 1.0
    return ret


def mtf_diffraction_limit_cutoff(wvl_um, NA):
    """
    :param wvl_um: wavelength
    :param NA:
    :return: MTF cutoff frequency in cy/mm
    """
    cutoff = 2.0 * NA / (wvl_um/1000.0)
    return cutoff


def mtf_diffraction_limit(wvl_um, NA, u_cymm):
    """
    computes diffraction limit (incoherent) MTF
    :param wvl_um: wavelength in um
    :param NA: Numerical Aperture
    :param u_cymm: spatial frequency in cy/mm as numpy array
    :return: numpy array of MTF values
    """
    k = (wvl_um / 1000.0) / (2.0 * NA)
    phi = np.arccos(k*u_cymm)
    mtf = (2 / math.pi) * np.subtract(phi, np.multiply(np.cos(phi), np.sin(phi)))
    return mtf


class MBeam:
    """
    Gaussian beam calculation including M^2

    P. A. Belanger, Optics Letters, 16, 4, Feb 15,1991. Beam propagation and the ABCD ray matrices
    define q by:  1/q = 1/R -i ((M^2*Lambda)/(pi*W^2)) or in other words just substitute M2*Lambda for Lambda
    """

    # define a new constant self._wvl_M2 and use that for calculation so that self.wvl is correct if
    # returned.  Test against Zemax for M = 1 and M != 1

    def __init__(self, wvl, omega, cv=0.0, M=1.0, rindex=1.0):
        """
        :param wvl: wavelength in lens units
        :param omega: exp(-2) radius of irradiance pattern (of embedded TEM00 for M != 1.0)
        :param cv: 1/radius of the phase front (0.0 for waist)
        :param M: beam quality parameter (not M^2, just M)
        :param rindex: refractive index
        """
        self._wvl_M2 = wvl * math.pow(M, 2)
        self.M = M
        self.wvl = wvl
        self.q = 1.0 / complex(cv, -self.wvl / (math.pi * rindex * math.pow(omega, 2)))


    def cv(self):  # 1/R
        """
        :return: curvature of phase front
        """
        temp = 1.0 / self.q
        return temp.real


    def omega(self, rindex=1.0):
        """
        :param rindex: refractive index
        :return: 1/e^2 spot semi-diameter of mixed mode beam with quality M or TEM00 spot with M = 1
        """
        return math.sqrt(-self._wvl_M2 / (math.pi * rindex * (1.0 / self.q).imag))


    def z0(self):  # distance to waist
        """
        :return: distance to waist
        """
        return -self.q.real

    def omega0(self, rindex=1.0):
        """
        :param rindex: refractive index
        :return: exp(-2) irradiance semi-diameter of mixed mode beam waist from beam with quality M or TEM00 waist with M = 1
        """
        qwaist = self.q + self.z0()
        return math.sqrt(-self._wvl_M2 / (math.pi * rindex * (1.0 / qwaist).imag))


    def half_angle_divergence(self):
        """
        :return: half angle of the far field divergence in radians of mixed mode beam with quality M or TEM00 with M = 1
        """
        return 2.0 * self._wvl_M2 / (math.pi * 2 * self.omega0())


    def confocal_parameter(self):
        """
        :return: confocal parameter
        """
        return 2.0 * math.pi * math.pow(self.omega0(), 2) / self._wvl_M2

    def Rayleigh_range(self):
        """
        :return: Rayleigh range
        """
        return math.pi * math.pow(self.omega0(), 2) / self._wvl_M2

    def glass(self, z):  # propagate thru z dist
        """
        Propagates beam through glass
        :param z: propagation distance
        """
        self.q = self.q + z


    def air(self, z):
        """
        Propagates beam through air
        :param z:propagation distance
        """
        self.q = self.q + z


    def propagate(self, a, b, c, d):
        """
        ABCD matrix propagation of beam
        :param a: ABCD component
        :param b: ABCD component
        :param c: ABCD component
        :param d: ABCD component
        """
        self.q = (a * self.q + b) / (c * self.q + d)


    def thin_lens(self, efl):  # propagate through thin lens
        """
        propagation through thin lens
        :param efl: effective focal length
        """
        a = 1.0
        b = 0
        c = -1.0 / efl
        d = 1.0
        self.propagate(a, b, c, d)


    def surface(self, cv, n, nprime):
        """
        beam propagation through optical surface
        :param cv: surface curvature (lens design sign convention)
        :param n: refractive index prior to surface
        :param nprime: refractive index following surface
        """
        a = 1.0
        b = 0
        c = (nprime - n) * (-cv) / nprime
        d = n / nprime
        self.propagate(a, b, c, d)


    def element(self, n, cv1, thick, nglass, cv2, nprime):
        """
        Beam propagation through thick lens
        :param n: refractive index prior to S1
        :param cv1: curvature of S1 (lens design sign convention)
        :param thick: center thickness of lens element
        :param nglass: refractive index of material
        :param cv2: curvature of S2
        :param nprime: refractive index following S2
        """
        self.surface(cv1, n, nglass)
        self.glass(thick)
        self.surface(cv2, nglass, nprime)
