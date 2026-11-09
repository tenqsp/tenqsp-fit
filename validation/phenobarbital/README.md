# Phenobarbital: reference results

- `reference_population_nlme.csv`, `reference_population_nlme_m2ll.txt`: R's `nlme`, from
  `reference.R` (`Rscript validation/phenobarbital/reference.R`).
- `reference_population_nonmem.csv`: NONMEM 7, FOCE with interaction, the phenobarbital
  example run distributed with Pharmpy (`pheno.mod`, `pheno.ext`); the same model and
  data. Variances as NONMEM reports them (OMEGA, SIGMA); standard errors from NONMEM's
  default covariance step (the sandwich estimate). −2LL with the constant: 871.147002.
