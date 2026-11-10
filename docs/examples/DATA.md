# The example data

The measured data in this folder come from published studies and are copied, unchanged, from
the software packages that distribute them. They are not part of tenqsp-fit, and the
project's Apache-2.0 licence does not cover them. Cite the original studies when you use
them. The simulated data (`timekill.csv`, made by `make_timekill.py`) are the project's own.

## theoph.csv: theophylline

Twelve subjects, an oral dose of theophylline (`Dose`, mg/kg) and eleven plasma
concentrations each (`conc`, mg/L) over 24 hours (`Time`); body weight `Wt` (kg).

- Study: Boeckmann AJ, Sheiner LB, Beal SL. *NONMEM Users Guide, Part V*. NONMEM Project
  Group, University of California, San Francisco; 1994.
- Copied from: R's `datasets` package, `Theoph`.

## phenobarb.csv: phenobarbital in newborns

Fifty-nine preterm newborns, phenobarbital given intravenously for seizure prevention,
measured during routine care: a few concentrations per infant. One row per dose or
measurement: `Subject`, birth weight `Wt` (kg), 5-minute Apgar score `Apgar` and whether it
is at least 5 (`ApgarInd`), `time` (h), `dose` (µg/kg), `conc` (µg/L), as the source gives them.

- Study: Grasela TH Jr, Donn SM. Neonatal population pharmacokinetics of phenobarbital
  derived from routine clinical data. *Dev Pharmacol Ther*. 1985;8(6):374–383.
- Copied from: R's `nlme` package, `Phenobarb`.

## warfarin.csv: warfarin concentrations and clotting response

Thirty-two subjects, one oral dose of warfarin (`amt`, mg) at time 0, then plasma warfarin
concentrations (`dvid` 1, mg/L) and the prothrombin complex activity (`dvid` 2, PCA, % of
normal) over up to 144 hours (`time`); body weight `wt` (kg), `sex` and `age` (years) as in
the source.

- Studies: O'Reilly RA, Aggeler PM, Leong LS. Studies on the coumarin anticoagulant drugs:
  the pharmacodynamics of warfarin in man. *J Clin Invest*. 1963;42(10):1542–1551.
  O'Reilly RA, Aggeler PM. Studies on coumarin anticoagulant drugs: initiation of warfarin
  therapy without a loading dose. *Circulation*. 1968;38(1):169–177.
- Copied from: the warfarin demo data distributed with Monolix, as included in the R
  package `monolix2rx` (`cov/data/warfarin_data.csv`). Missing values, written `.` there,
  are empty here.
