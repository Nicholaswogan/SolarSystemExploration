# CIA

Following are the origin of the CIA cross sections in this folder. Note one minor issue for H2-H2, H2-He, and N2-N2 cross sections. The cross sections only extend to 250 microns, but in reality HITRAN data says their is absorption beyond 250 microns. This is a minor problem for cold planets (< ~250 K) which emit a small amount of energy beyond 250 microns.

- **H2-H2**: From petitRADTRANS database \[1\]. See their Table 3. The cross sections are a compilation from many sources, including HITRAN \[2\].

- **H2-He**: From petitRADTRANS database \[1\]. See their Table 3. The cross sections are a compilation from many sources, including HITRAN \[2\].

- **N2-N2**: From petitRADTRANS database \[1\]. From HITRAN \[2\], but there are possibly other sources because the data do not perfectly match HITRAN.

- **CH4-CH4**: Directly from HITRAN \[2\]. Data came in regular grid.
  
- **N2-O2**: Directly from HITRAN \[2\]. Interpolated to regular grid in complicated way.
  
- **O2-O2**: Directly from HITRAN \[2\]. Interpolated to regular grid in complicated way.
  
- **H2-CH4**: Directly from HITRAN \[2\]. Data came in regular grid.

- **CO2-CO2**: Directly from HITRAN \[2\]. Interpolated to regular grid in complicated way.
  
- **CO2-CH4**: Directly from HITRAN \[2\]. Data came in regular grid.
  
- **CO2-H2**: Directly from HITRAN \[2\]. Data came in regular grid.
  
- **N2-H2**: Directly from HITRAN \[2\]. Interpolated to regular grid in simple way.

- **H2O-H2O**: CIA formulation of the MT_CKD v3.5 H2O continuum \[3\].
  
- **H2O-N2**: CIA formulation of the MT_CKD v3.5 foreign continuum \[3\]. Assumes N2 is the foreign species.

<!-- @ARTICLE{2019Icar..328..160K,
       author = {{Karman}, Tijs and {Gordon}, Iouli E. and {van der Avoird}, Ad and {Baranov}, Yury I. and {Boulet}, Christian and {Drouin}, Brian J. and {Groenenboom}, Gerrit C. and {Gustafsson}, Magnus and {Hartmann}, Jean-Michel and {Kurucz}, Robert L. and {Rothman}, Laurence S. and {Sun}, Kang and {Sung}, Keeyoon and {Thalman}, Ryan and {Tran}, Ha and {Wishnow}, Edward H. and {Wordsworth}, Robin and {Vigasin}, Andrey A. and {Volkamer}, Rainer and {van der Zande}, Wim J.},
        title = "{Update of the HITRAN collision-induced absorption section}",
      journal = {\icarus},
         year = 2019,
        month = aug,
       volume = {328},
        pages = {160-175},
          doi = {10.1016/j.icarus.2019.02.034},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2019Icar..328..160K},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
} -->



<!-- Analysis of water vapor absorption in the far-infrared and submillimeter regions using surface radiometric measurements from extremely dry locations. -->



## References
1. [https://doi.org/10.1051/0004-6361/201935470](https://doi.org/10.1051/0004-6361/201935470)
2. [https://hitran.org/cia/](https://hitran.org/cia/)
3. [https://hitran.org/mtckd/](https://hitran.org/mtckd/)