# kdistributions

This subdirectory contains k-coefficients that I computed with HELIOS-K. I have a fork of HELIOS-K which I used to compute all these opacities: https://github.com/Nicholaswogan/HELIOS-K/tree/8f56e8a507d8f68db5eac1b146f9a2b81da52fe5

All species are computed for pressures between ~10^-5 and 10^3 bars and temperatures between 100 and 2000 K. The only exception is O3, which only computes temperatures up to 1000 K. In general, these k-coefficients are designed to be applicable to terrestrial planetary atmospheres < 2000 K in the habitable zone. 

Below are a summary of the relevant settings I used in HELIOS-K. These settings are largely motivated by the k-coefficients used in the `src.n68equiv` version of [ExoRT](https://github.com/storyofthewolf/ExoRT), and discussions with Eric Wolf.

## H2O
- I used HITEMP 2010 data for 0 to 30000 cm^-1, and HITRAN 2016 data for 30000 to 42000 cm^-1
- 25 cm^-1 cutoff
- Pressure broadening is done by air (qalphaL = 0.0)
- Voigt lineshape (profile = 1)
- Plinth or base is removed (removePlinth = 1). This setting is unique to H2O, because it is combined with continuum opacities for H2O. See DOI: 10.1029/2010JD015505.

## CO2
- I used HITEMP 2010 data
- 500 cm^-1 cutoff
- Pressure broadening is done by itself (qalphaL = 1.0)
- sub-Lorentzian lineshape with Perrin and Hartmann (1989) chi factors used for all wavenumber bins

## CH4
- I used HITEMP 2020 data
- 25 cm^-1 cutoff
- Pressure broadening is done by air (qalphaL = 0.0)
- Voigt lineshape (profile = 1)

## CO
- I used HITEMP 2019 data
- 25 cm^-1 cutoff
- Pressure broadening is done by itself (qalphaL = 1.0)
- Voigt lineshape (profile = 1)

## O2
- I used HITRAN 2016 data
- 25 cm^-1 cutoff
- Pressure broadening is done by air (qalphaL = 0.0)
- Voigt lineshape (profile = 1)

## O3
- I used HITRAN 2016 data
- 25 cm^-1 cutoff
- Pressure broadening is done by air (qalphaL = 0.0)
- Voigt lineshape (profile = 1)

## NH3
- I used HITRAN 2016 data
- 25 cm^-1 cutoff
- Pressure broadening is done by air (qalphaL = 0.0)
- Voigt lineshape (profile = 1)

## SO2
- I used HITRAN 2016 data
- 25 cm^-1 cutoff
- Pressure broadening is done by air (qalphaL = 0.0)
- Voigt lineshape (profile = 1)

## C2H2
- I used HITRAN data downloaded on 4/23/25 with `hitran-api`
- 25 cm^-1 cutoff
- Pressure broadening is done by air (qalphaL = 0.0)
- Voigt lineshape (profile = 1)

## C2H6
- I used HITRAN data downloaded on 4/23/25 with `hitran-api`
- 25 cm^-1 cutoff
- Pressure broadening is done by air (qalphaL = 0.0)
- Voigt lineshape (profile = 1)

## HCl
- I used HITRAN data downloaded on 4/23/25 with `hitran-api`
- 25 cm^-1 cutoff
- Pressure broadening is done by air (qalphaL = 0.0)
- Voigt lineshape (profile = 1)

## N2O
- I used HITRAN data downloaded on 4/23/25 with `hitran-api`
- 25 cm^-1 cutoff
- Pressure broadening is done by air (qalphaL = 0.0)
- Voigt lineshape (profile = 1)

## OCS
- I used HITRAN data downloaded on 4/23/25 with `hitran-api`
- 25 cm^-1 cutoff
- Pressure broadening is done by air (qalphaL = 0.0)
- Voigt lineshape (profile = 1)