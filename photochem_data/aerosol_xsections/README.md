
# Aerosol optical properties

I use the `miepython` python package to compute aerosol optical properties: https://github.com/scottprahl/miepython

**Folder and File formats**

Each folder contains a file with the name `mie_<folder-name>.h5` that has optical properties for a kind of aerosol. The file has the following structure

```yaml
wavelengths: Wavelength [nm]
radii: Particle radii [um]
w0: Single scattering albedo [unitless], dimensions (len(radii),len(wavelengths))
qext: Extinction [1/particle], dimensions (len(radii),len(wavelengths))
g0: Asymmetry factor [unitless], dimensions (len(radii),len(wavelengths))
```

## Folders

`khare1984` is data for optical properties of fractal hydrocarbon aerosols from [Khare et al. (1984)](https://www.sciencedirect.com/science/article/pii/0019103584901428) downloaded from [hitran.org](hitran.org).

`palmer1975` is data for optical properties of sulfuric acid aerosols (95.6% H$_2$SO$_4$ and 4.4% H$_2$O) from [Palmer and Williams (1975)](https://doi.org/10.1364/AO.14.000208) downloaded from [hitran.org](hitran.org). We

