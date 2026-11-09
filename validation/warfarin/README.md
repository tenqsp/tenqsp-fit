# Warfarin: reference results

Monolix's fits of the warfarin concentrations (`dvid` 1), as distributed with the R package
`monolix2rx` (`cov/w1` and `cov/w3`, projects `warfarin_covariate1_project.mlxtran` and
`warfarin_covariate3_project.mlxtran`), with the data in `docs/examples/warfarin.csv`:

- `reference_monolix_base.csv`: one compartment, first-order absorption with a lag time;
  Tlag, ka, V and Cl lognormal between subjects; error `combined1` (sd = a + b f).
  −2LL 657.358 (importance sampling).
- `reference_monolix_sex_on_V_CL.csv`: the same with sex on V and on Cl
  (`beta_*_tSex_F`, log scale, males the reference). −2LL 641.683 (importance sampling).

Estimates by SAEM; standard errors by stochastic approximation (`se_sa`). The error model
and the method differ from tenqsp-fit's (`combined`, sd = sqrt(a² + (b f)²); FOCE-I), so
the comparison is of the estimates, within their standard errors, not of −2LL.
