# %% [markdown]
# # Tutorial 3: phenobarbital in newborns, covariates from sparse data
#
# Phenobarbital given to 59 preterm newborns to prevent seizures, measured during routine
# care (R's `Phenobarb` data, from Grasela and Donn, 1985): many doses, and only a few blood
# samples per infant, taken when it suited the ward. It is the classic example of a
# population analysis of routine clinical data, and the example model of NONMEM, so its
# answers are known. This tutorial builds the model the way a modeller would:
#
# 1. the data: what sparse routine data look like;
# 2. a model without covariates;
# 3. clearance and volume in proportion to body weight;
# 4. a low Apgar score on the volume, and whether the data support it;
# 5. the final model against NONMEM's own run of it;
# 6. shrinkage: what the infants' own values can and cannot show with so few samples.
#
# The model is written in fit text here: the same statements as the Python calls of the
# first two tutorials.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tenqsp as tq

import tenqsp_fit as tf

data = pd.read_csv("../examples/phenobarb.csv")
# NONMEM's and R's fits of the final model, stored in the repository
REF = "../../validation/phenobarbital/"
ref_nonmem = pd.read_csv(REF + "reference_population_nonmem.csv").set_index("quantity")
ref_nlme = pd.read_csv(REF + "reference_population_nlme.csv").set_index("quantity")
print(data.head(8))

# %% [markdown]
# ## 1. The data
#
# One row per dose (`dose`) or per sample (`conc`); birth weight `Wt` (kg) and the 5-minute
# Apgar score `Apgar` are columns. Each infant gets a loading dose and then maintenance doses
# every 12 hours or so; the samples fall wherever they fell.

# %%
doses, samples = data[data.dose.notna()], data[data.conc.notna()]
per_infant = samples.groupby("Subject").size()
print(f"{data.Subject.nunique()} infants, {len(doses)} doses, {len(samples)} samples")
print(f"samples per infant: {per_infant.value_counts().sort_index().to_dict()}")
print(f"birth weight {data.Wt.min()} to {data.Wt.max()} kg; "
      f"Apgar below 5 in {data[data.Apgar < 5].Subject.nunique()} infants")  # fmt: skip

# an infant with one sample, one with three, and the one with the most
show = [per_infant[per_infant == 1].index[0], per_infant[per_infant == 3].index[0],
        per_infant.idxmax()]  # fmt: skip
fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
for ax, s in zip(axes, show, strict=True):
    d = data[data.Subject == s]
    ax.plot(d.time, d.conc, "o", color="#1f3b73")
    for t in d[d.dose.notna()].time:
        ax.axvline(t, color="0.75", lw=0.8, zorder=0)
    ax.set_title(f"infant {s}: {d.Wt.iloc[0]} kg, Apgar {d.Apgar.iloc[0]}", fontsize=9)
    ax.set_xlabel("time (h)")
axes[0].set_ylabel("concentration")
assert per_infant.median() <= 3

# %% [markdown]
# Grey lines are doses. Most infants have two or three samples, some only one: enough to
# place each infant's clearance and volume only roughly. The
# population model is what makes these data usable at all.
#
# ## 2. A model without covariates
#
# One compartment, the doses given intravenously into it. Clearance and volume vary between
# infants (lognormal), and the measurement error is proportional:

# %%
model = tq.Model.from_text("""
model pheno
param CL = 0.005, V = 1
d_t(centr) = -CL/V*centr
observable conc = centr/V
""")

HEAD = """fit pheno
data "../examples/phenobarb.csv" with ID=Subject, TIME=time, AMT=dose, DV=conc
unit ID
doses into centr
observe conc = DV
"""
TAIL = """error conc ~ proportional(0.1)
method laplace
options progress=none
"""


def fit(estimates):
    """The fit text: the lines common to every model here, and its own estimates."""
    return tf.fit(tf.Problem.from_text(HEAD + estimates + TAIL, model=model))


base = fit("""
estimate CL ~ lognormal(median=0.007, sdlog=0.3)
estimate V ~ lognormal(median=1.4, sdlog=0.3)
""")
print(base)

# %% [markdown]
# The `data` line maps the file's own column names to the standard ones (as NONMEM's
# `$INPUT`): rows with a `dose` are doses, rows with a `conc` are measurements, and
# `doses into centr` says where the doses go.
#
# Clearance varies between infants with a coefficient of variation of about 34 %, the
# volume about 47 %.
#
# ## 3. Body weight
#
# Newborns differ in size more than adults do: here by a factor of six. Clearance and volume
# in proportion to birth weight, `CL = CL_kg·Wt` and `V = V_kg·Wt`, are written as the
# typical values (medians) of the two distributions:

# %%
WEIGHT = """estimate CL_kg = 0.005
estimate V_kg = 1
estimate CL ~ lognormal(median=CL_kg*Wt, sdlog=0.2)
"""
weight = fit(WEIGHT + "estimate V ~ lognormal(median=V_kg*Wt, sdlog=0.2)\n")
print(weight)
print(f"-2LL {base.m2ll:.2f} -> {weight.m2ll:.2f}: lower by {base.m2ll - weight.m2ll:.1f}")
assert base.m2ll - weight.m2ll > 100
assert weight.estimates["CL.sdlog"] < 0.6 * base.estimates["CL.sdlog"]
assert weight.estimates["V.sdlog"] < 0.5 * base.estimates["V.sdlog"]

# %% [markdown]
# −2LL falls by about 140, for no extra quantity estimated (`CL_kg` and `V_kg` replace the
# two medians), and the variation left between infants roughly halves for clearance and
# falls by more than half for the volume: most of what differed between infants was their
# size.
#
# ## 4. A low Apgar score
#
# The 5-minute Apgar score rates a newborn's condition from 0 to 10. Infants who scored
# below 5 may hold more water, so the volume gets a factor `1 + apgar` for them, a conditional
# formula of the `Apgar` column:

# %%
APGAR = """estimate apgar = 0.1 in (-0.99, 5)
estimate V ~ lognormal(median=V_kg*Wt*((1 + apgar) if Apgar < 5 else 1), sdlog=0.2)
"""
final = fit(WEIGHT + APGAR)
print(final)
drop = weight.m2ll - final.m2ll
print(f"-2LL lower by {drop:.2f} for one more quantity (the 5 % cut-off is 3.84)")
assert 3.84 < drop < 6

# %% [markdown]
# One more quantity lowers −2LL by 4.5, past the cut-off of 3.84 (a likelihood ratio test,
# chi-square with one degree of freedom, at 5 %). The volume is about 16 % larger after a
# low Apgar score, and its 95 % interval only just excludes 0: these data support the effect,
# but not strongly.
#
# ## 5. The same answer as NONMEM
#
# The final model is NONMEM's phenobarbital example, and its run (FOCE with interaction,
# the method of `laplace`) is stored with the tutorials. NONMEM reports variances; the
# table converts:

# %%
e = final.estimates
mine = {
    "CL_kg": e["CL_kg"], "V_kg": e["V_kg"], "apgar": e["apgar"],
    "CL.var": e["CL.sdlog"] ** 2, "V.var": e["V.sdlog"] ** 2,
    "conc.proportional.var": e["conc.proportional"] ** 2,
}  # fmt: skip
compare = pd.DataFrame({"tenqsp-fit": mine, "NONMEM": ref_nonmem.estimate})
compare["ratio"] = compare["tenqsp-fit"] / compare["NONMEM"]
print(compare.round(6).to_string())
print(f"-2LL: tenqsp-fit {final.m2ll:.4f}, NONMEM 871.1470")
assert abs(final.m2ll - 871.147) < 0.01
assert np.allclose(compare["ratio"], 1, atol=2e-3)

# %% [markdown]
# Every value agrees to three or four significant digits, and −2LL to the fourth decimal.
# R's `nlme` (a different approximation of the same likelihood) lands close by, at −2LL
# 870.89. The standard errors differ more (the volume's 0.028 here, 0.027 in NONMEM; the
# Apgar effect's 0.078 and 0.084): NONMEM's default covariance step is the "sandwich"
# estimate, and tenqsp-fit's is the curvature of −2LL alone. Both are approximations, and
# the difference is small next to the estimates' own uncertainty.

# %%
row = final.table().set_index("quantity")
for q in ("V_kg", "apgar"):
    print(f"{q}: SE {row.loc[q, 'SE']:.4f} here, {ref_nonmem.loc[q, 'se']:.4f} in NONMEM")
assert abs(final.m2ll - float(open(REF + "reference_population_nlme_m2ll.txt").read())) < 1
assert np.isclose(e["V_kg"], ref_nlme.loc["V_kg", "estimate"], rtol=0.02)

# %% [markdown]
# ## 6. Shrinkage
#
# The doctor's warning above: the shrinkage of clearance is 47 %. Each infant's own
# clearance (`final.individual()`, NONMEM's EBEs) is a compromise between its data and the
# typical value, and with two or three samples the data say little about clearance on
# their own. So the infants' values are pulled towards the
# typical value: their spread is about half the spread the model estimated.

# %%
ind = final.individual().set_index("unit")
wt = data.groupby("Subject").Wt.first()
eta = np.log(ind["CL"] / (e["CL_kg"] * wt.loc[ind.index]))
print(f"spread of the infants' own log CL around CL_kg*Wt: {eta.std():.3f}; "
      f"estimated between infants: {e['CL.sdlog']:.3f}")  # fmt: skip
print("shrinkage:", {k: round(v, 3) for k, v in final.shrinkage.items()})
assert eta.std() < 0.7 * e["CL.sdlog"]
assert final.shrinkage["CL"] > 0.3 > final.shrinkage["V"]

fig, ax = plt.subplots(figsize=(5, 3.5))
ax.plot(wt.loc[ind.index], ind["CL"], "o", ms=4, color="#1f3b73", label="each infant's own CL")
w = np.linspace(wt.min(), wt.max(), 2)
ax.plot(w, e["CL_kg"] * w, "k-", lw=1, label="CL_kg·Wt")
ax.set_xlabel("birth weight (kg)")
ax.set_ylabel("clearance")
ax.legend(fontsize="small")

# %% [markdown]
# The infants' clearances sit close to the line, closer than the infants really do: a plot
# like this one, or of the infants' values against another covariate, would understate how
# much infants differ and could hide (or invent) a covariate effect. With shrinkage above
# 30 % the doctor says so, and the model is better judged by fits to the data themselves.
# The volume has little shrinkage (13 %): the concentrations determine it better.

# %%
fig = final.plot_gof()

# %% [markdown]
# And the three infants of section 1, now with their fitted curves: the population model
# fills in, between two or three samples, a concentration that rises with each dose and
# falls between doses.

# %%
fig = final.plot_individual(units=show)

# %% [markdown]
# ## What this shows
#
# - Sparse routine data (two or three samples per infant) still determine the typical
#   clearance and volume to a few percent, once the infants are fitted together.
# - Body weight explains most of the differences between infants; a low Apgar score adds a
#   small effect on the volume that the likelihood ratio test only just supports.
# - `laplace` reproduces NONMEM's FOCE-I run of the same model: estimates to three or four
#   digits, −2LL to the fourth decimal.
# - With high shrinkage, the infants' own values are drawn to the typical value; judge the
#   model by its fit to the data, not by plots of those values.
