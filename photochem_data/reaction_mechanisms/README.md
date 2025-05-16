`zahnle_earth.yaml` is Kevin Zahnle's network of reactions. He calls the network `earth_125.rx`. The meaning of "125" is unclear. I have documented the small changes to the network [at this Github repo.](https://github.com/Nicholaswogan/ImpactAtmosphere/blob/main/ImpactAtmosphere/data/notes.md). But from now, onward, I will document the changes here.

# 10/27/21

- Several species had valid polynomial Gibbs energy fits down to 100 K. But this was not compatible with Titan, because Titan is really cold. Therefore, I decreased the lower bounds of the fits to 50 K.
- I added many particles. Particles are made in various way from species. 

I added

```yaml
- equation: C2H4 + N2D <=> CH3CN + H
  rate-constant: {A: 2.3e-10, b: 0.0, Ea: 503}
```

I was modeling Titan, and was getting far to little CH3CN. Lavvas et al. (2008) uses this reaction rate.

I will keep the reaction for now, but note that this reaction might not be correct. From Loison et al (2015),

> The CH3CN molecule is not produced in our model by the N(2D) + C2H4 reaction but through the association reaction H+CH2CN (Balucani et al., 2012; Lee et al., 2011). We calculated the rate constant from our semi-empirical model for association reactions (Hébrard et al., 2013), the CH2CN radical being produced by the H + C2H4CN reaction (this work, C2H4CN itself being the product of the H + C2H3CN association) and the N + C2H3 reaction (Payne et al., 1996).

I would like to follow Loison et al.'s advice, but our model is missing two things

1. Radiative association reactions: A + B => AB + hv. These reactions are important for some speices in Titan's atmosphere (Vuitton et al. 2012)
2. We do not have C2H4CN in the network.

# 12/16/21

I added

```yaml
- equation: Cl + O3 <=> ClO + O2
  rate-constant: {A: 2.8e-11, b: 0.0, Ea: 250}
```

From Atkinson et al. (2007). Full citation:

> R. Atkinson, D. L. Baulch, R. A. Cox, J. N. Crowley, R. F. Hampson, et al.. Evaluated kinetic and photochemical data for atmospheric chemistry: Volume III ? reactions of inorganic halogens. Atmospheric Chemistry and Physics Discussions, European Geosciences Union, 2006, 6 (2), pp.2281-2702. ⟨hal-00301101⟩

# 12/23/21

I updated the following reaction. Rate from JPL-15

```yaml
- equation: S + O2 <=> SO + O
  rate-constant: {A: 1.6e-12, b: 0, Ea: -100.0}
```

Also added NH3aer and He.

# 3/9/23

I updated the following reaction following [Lovejoy et al. (1996)](https://doi.org/10.1021/jp962414d)

```yaml
# - equation: SO3 + H2O (+ M) <=> H2SO4 (+ M)
#   type: falloff
#   low-P-rate-constant: {A: 9.0e-29, b: -2.0, Ea: -1000.0}
#   high-P-rate-constant: {A: 3.6e-10, b: -2.0, Ea: -1000.0}
- equation: SO3 + H2O + H2O <=> H2SO4 + H2O
  rate-constant: {A: 2.3e-43, b: 1.0, Ea: -6540.0}
```

# 10/27/23

I updated a variety of thermodynamics for CH3O and H2COH, and rates for related species. All changes were motivated by [Xu et al. (2015)](https://doi.org/10.1021/acs.jpca.5b00553). Details for these changes are justified at this github repository: https://github.com/Nicholaswogan/RateExplorer/tree/main/experiments/10-02-23

I also updated thermodynamics for H2 and Cl2 following NIST, to include thermodynamic data up to 6000 K.

# 1/16/24

I added two new photolysis branches. All the H2O branches are now OH + H, H2 + O1D, and O + H + H. Branching ratios are based on JPL-19. I also added H2 photolysis, taking data from the Leiden database. 

Finally, I also, added gas-phase S8 into the code, along with 3 reactions connecting S8 to the rest of the network. These reactions + thermodynamics come from Kevin's original network he sent me (https://github.com/Nicholaswogan/ImpactAtmosphere/blob/main/ImpactAtmosphere/data). I also added S8 photolysis, but only as an opacity source (S8 absorbs light, but does not photolyze). I also added the particles S2aer and S8aer, with saturation vapor pressures based on Zahnle et al. (2016) (10.3847/0004-637X/824/2/137).

# 5/23/24

First, I want to simply state the reactions that I have changed since Kevin sent me `earth_125_5-26-2021_modified.rx`. Note that `earth_125_5-26-2021.rx` had a couple duplicates and bad mass balance reactions, so Kevin and I fixed those to make `earth_125_5-26-2021_modified.rx` (see notes here: https://github.com/Nicholaswogan/ImpactAtmosphere/blob/main/ImpactAtmosphere/data/notes.md). Since, `earth_125_5-26-2021_modified.rx`, I have made the following changes to the kinetics

- H2CO + H <=> HCO + H2 (10/27/23)
- CH3 + O <=> H2CO + H (10/27/23)
- CH3 + O <=> HCO + H2 (10/27/23)
- H + H2CO (+ M) <=> CH3O (+ M) (10/27/23)
- H + H2CO (+ M) <=> H2COH (+ M) (10/27/23)
- S + O2 <=> SO + O (altered rate, 12/23/21)
- SO3 + H2O + H2O <=> H2SO4 + H2O (altered rate, 3/9/23)
- Cl + O3 <=> ClO + O2 (added, 12/16/21)
- C2H4 + N2D <=> CH3CN + H (added, 10/27/21)

The first 5 rates were updated following Xu et al. (2015) as explained in note 10/27/23. They are great and should stay.

I took a close look at the next 4 rates, to make sure I didn't introduce anything that would break down at high or low temperatures. My analysis is detailed here: https://github.com/Nicholaswogan/RateExplorer/tree/main/experiments/5-29-24 . In summary, I'm going to keep all the rates, except S + O2 <=> SO + O, which I am updating to the following based on [Lu et al. (2004)](http://dx.doi.org/10.1063/1.1792611). The rate below is a custom fit, designed to work well at high and low temps.

```yaml
- equation: S + O2 <=> SO + O
  rate-constant: {A: 3.21e-16, b: 1.35, Ea: -285.8}
```

Also, I have updated the rate of HO2 + HO2. Kevin told me via email that his original rate was a typo. This new rate is described here: https://github.com/Nicholaswogan/RateExplorer/tree/main/experiments/4-18-24 . It is a custom fit based on [Klippenstein et al. (2022)](https://doi.org/10.1016/j.combustflame.2021.111975) and [Kircher and Sander (1984)](https://doi.org/10.1021/j150654a029).

```yaml
- equation: HO2 + HO2 <=> H2O2 + O2
  rate-constant: {A: 3.19e-19, b: 2.01, Ea: -1274.3}
```

# 5/30/24

Here, I update all the gas saturation vapor pressures. All updates are cited and detailed here: https://github.com/Nicholaswogan/RateExplorer/tree/main/saturation_thermo commit c6f7135da873e4ef9be4944a2511dbadc3299ee8.

I use a new scheme based on the Clausius Clapeyron equation, which assumes latent heat has linear temperature dependence. The latent heat is allowed a discontinuous transition between ice and liquid phases.

I have also pieced together Shomate polynomials for each condensible, based on the saturation vapor pressure and the gas-phase gibbs energy. I've saved all these polynomials in a separate file, which are useful for equilibrium chemistry calculations.
