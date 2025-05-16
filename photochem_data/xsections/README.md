
# Cross Section Data

This directory contains cross section data for photolysis reactions. Citations for the data are in `metadata.yaml` and `bib.bib`. The structure of each hdf5 file is as follows. Here I use `CO2.h5` as an example.

```yaml
wavelengths: Wavelengths in nm
photoabsorption: Total photoabsorption cross section in cm^2/molecule
photodissociation: Photodissociation cross section in cm^2/molecule
photoionisation: Photoionization cross section in cm^2/molecule
photodissociation-qy:
  wavelengths: Wavelengths for which quantum yields are defined
  CO2 + hv => CO + O: Quantum yields for the CO + O branch
  CO2 + hv => CO + O1D: Quantum yields for the CO + O1D branch
```

This data assumes that cross sections are zero beyond the edges of the specified wavelengths. Quantum yields, on the other hand, should be constantly extrapolated.

`bins.h5` has the single key `wavl`, which gives the wavelength bins in nm. These bins were designed by Kevin Zahnle and Jim Kasting long ago.

# Notes

## 9/7/21

Kevin gave me his list of photolysis reactions, cross section data, and quantum yield data. All are located in the folder `Zahnle_Kevin_data`.

Here, I implemented Kevin Zahnle's cross sections for most species. Here are some of Kevin's notes on this cross sections.

- *C4H2* xs is 2 times C2H2 xs.
- *HSO* set to HO2. Kevin says "HSO is more stable than HO2, but, you know, photolysis never really matters for these short-lived species"
- *S8L*. Estimate for cross section of linear S8 chains in visible. set equal to S8R in uv
- *CH3OH*. CH3OH + hv  => CH3O +  H this branch is reported 86±10%. The other processes are undetected
- *CH3CN*. CH3CN + HV  -> CH3 +  CN the H + CH2CN is in fact somewhat more important

I did not use Kevin's cross sections for the following species: 'CO2', 'H2CO', 'O2', 'O3', 'SO2', and 'NO'. Instead I use the Phidrates database.

I omit the following photolysis reactions

```yaml
- equation: SO2 + hv => S + O + O
  type: photolysis
- equation: HNO + hv => NO + H
  rate-constant:
    A: 1.0e-3
    b: 0.0
    Ea: 0.0
- equation: HNO2 + hv => NO + OH
  rate-constant:
    A: 1.0e-3
    b: 0.0
    Ea: 0.0
```

I omit `SO2 + hv => S + O + O` because Phidrates has no data for this reaction. I omit `HNO + hv => NO + H` and `HNO2 + hv => NO + OH` because Kevin hard-codes rates instead of using cross sections. I am unable to find cross sections. 

## 10/3/21

I noticed that when modeling Mars, CO2 photolysis was much higher than previous models. It looks like the Phidrates cross sections are too large for CO2 + hv => CO + O branch. So I switched to Kevin Zahnle's CO2 cross sections. NOW i use Kevin's cross sections for all but

'H2CO', 'O2', 'O3', 'SO2', 'NO'

Which uses Phidrates.

## 10/5/21

I added `metadata.yaml`, which contains citations for all the cross sections. I also added `check_consistency.py` which check for consistency between data and metadata.

## 1/16/24

- I updated photolysis branching ratios for H2O following JPL-19, and VULCAN
- Added H2 photolysis. 
- Changed S8 photolysis to create itself (i.e. absorption, but not photolysis)
- Deleted S8L, because it is not used

## 5/29/24

No new updates, I want to correct past notes. Here are a summary of the citations for each cross section. All are from Kevin Zahnle except,

Phidrates: HCCCN, O3, H2CO, O2, SO2, NO
Fahr and Nayak (1996): C4H4
Heays et al. (2017): H2

I added HCCCN and C4H4 photolysis a long while ago, but didn't note it. For C4H4, I followed [Lavvas et al. (2008)](https://doi.org/10.1016/j.pss.2007.05.026), using cross sections from Fahr and Nayak (1996), and yields from Gladstone et al. (1996). For HCCCN, I just took values from Phidrates.

The previous edit (1/16/24) made it so that when S8 rings only absorb radiation, but do not photolyze. The idea is that the rings would be split, but then quickly reform.

## 9/26/24

I updated all the data. Reproducing all this new data can be done with the scripts at https://github.com/Nicholaswogan/RateExplorer/tree/main/xsections at commit ef04d47be9b48825f6f2c6e7a242d94146143ac5 . Data mostly comes from the [Leiden database](https://home.strw.leidenuniv.nl/~ewine/photo/cross_sections.html), [Phidrates](https://phidrates.space.swri.edu/#) and [JPL-19](https://jpldataeval.jpl.nasa.gov/), but there are other sources as well all detailed in `metadata.yaml` and `bib.bib`

I've now separated absorption, dissociation and ionization. So, the model can account for absorption when it occurs without causing a reaction, which is important for species like SO2. Below is a list of the photolysis reactions that I removed:

```yaml
- C2H6 + hv => C2H5 + H
- CH3CHO + hv => CH3CO + H
- S8 + hv => S8
```

Here is a list of all the new reactions

```yaml
- C2H4 + hv => C2H2 + H + H
- C2H4 + hv => C2H2 + H2
- C2H6 + hv => C2H2 + H2 + H2
- C2H6 + hv => C2H4 + H + H
- C2H6 + hv => C2H4 + H2
- C2H6 + hv => CH3 + CH3
- C2H6 + hv => CH4 + 1CH2
- C3H6 + hv => C2H2 + CH4
- C3H6 + hv => C2H3 + CH3
- C3H6 + hv => C2H4 + 1CH2
- C3H6 + hv => C3H4 + H2
- C4H2 + hv => C2H + C2H
- C4H2 + hv => C2H2 + C2
- C4H2 + hv => C4H2
- CH4 + hv => CH2 + H + H
- CS + hv => C + S
- H2SO4 + hv => SO3 + H2O
- HNCO + hv => H + NCO
- HNCO + hv => NH + CO
- NH3 + hv => NH + H + H
- OClO + hv => ClO + O
- OClO + hv => ClO + O1D
- S4 + hv => S2 + S2
```