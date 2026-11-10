# %% [markdown]
# # Tutorial 5: warfarin, a covariate that is another covariate
#
# Warfarin in 32 volunteers, one oral dose each, then plasma concentrations over five days
# (the warfarin data of O'Reilly and colleagues, as distributed with Monolix). Monolix's
# demo fits it, and with sex on the volume it finds that women have a volume about 30 %
# smaller. This tutorial fits the same data and asks whether that is about sex:
#
# 1. the data, and who is in it;
# 2. the PK model, with a lag time, next to Monolix's estimates;
# 3. sex on the volume, as Monolix;
# 4. weight on the volume, and both together.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tenqsp as tq

import tenqsp_fit as tf

data = pd.read_csv("../examples/warfarin.csv")
# Monolix's fits of the same data, stored in the repository
REF = "../../validation/warfarin/"
ref_base = pd.read_csv(REF + "reference_monolix_base.csv").set_index("parameter")
ref_sex = pd.read_csv(REF + "reference_monolix_sex_on_V_CL.csv").set_index("parameter")
print(data.head(6))

# %% [markdown]
# ## 1. The data
#
# One row per dose (`amt`) or measurement (`dv`); `dvid` 1 is the warfarin concentration
# (mg/L), `dvid` 2 the clotting response (prothrombin complex activity), which this tutorial
# leaves out. Weight `wt` (kg), `sex` (1 male, 0 female) and `age` are columns.

# %%
people = data.drop_duplicates("id").set_index("id")
conc = data[data.dvid == 1]
print(f"{len(people)} subjects, {len(conc)} concentrations, one dose each "
      f"({people.amt.min():g} to {people.amt.max():g} mg)")  # fmt: skip
print(people.groupby("sex").wt.describe()[["count", "mean", "min", "max"]])
early = conc[conc.time < 24].id.nunique()
print(f"subjects sampled in the first 24 h: {early} of {len(people)}")

fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
for i, d in conc.groupby("id"):
    ax[0].plot(d.time, d.dv, "o-", ms=2.5, lw=0.7,
               color="#d62728" if people.sex[i] == 0 else "#1f3b73")  # fmt: skip
ax[0].set_xlabel("time (h)")
ax[0].set_ylabel("warfarin (mg/L)")
ax[0].set_title("red: women", fontsize=9)
ax[1].hist([people.wt[people.sex == 1], people.wt[people.sex == 0]], bins=10,
           color=["#1f3b73", "#d62728"], label=["men", "women"])  # fmt: skip
ax[1].set_xlabel("weight (kg)")
ax[1].legend(fontsize="small")
assert (people.sex == 0).sum() == 5

# %% [markdown]
# Five of the 32 are women, and they are lighter: about 51 kg on average, against 74 kg for
# the men. Keep that in mind. Most subjects were sampled from 24 hours on; only some in the
# first hours, when the drug is absorbed.
#
# ## 2. The PK model
#
# One compartment, absorption with a lag time: `lag depot = tlag` delays every dose into
# `depot` (NONMEM's ALAG). The lag time, ka, CL and V vary between subjects, as in Monolix.

# %%
model = tq.Model.from_text("""
model warfarin_pk
param ka = 1.5, CL = 0.13, V = 8, tlag = 0.8
d_t(depot) = -ka*depot
d_t(centr) = ka*depot - CL/V*centr
lag depot = tlag
observable conc = centr/V
""")

HEAD = """fit warfarin
data "../examples/warfarin.csv" with ID=id, TIME=time, AMT=amt, DV=dv
unit ID
doses into depot
observe conc = DV where dvid == 1
ignore DV where dvid == 2
estimate tlag ~ lognormal(median=0.8, sdlog=0.5)
estimate ka ~ lognormal(median=1.5, sdlog=0.5)
estimate CL ~ lognormal(median=0.13, sdlog=0.3)
"""
TAIL = """error conc ~ combined(0.3, 0.05)
method laplace
options progress=none
"""


def fit(volume):
    """The fit text: the lines common to every model here, and the volume's own lines."""
    return tf.fit(tf.Problem.from_text(HEAD + volume + TAIL, model=model))


base = fit("estimate V ~ lognormal(median=8, sdlog=0.2)\n")
print(base)

# %% [markdown]
# `ignore DV where dvid == 2` leaves the clotting response out of the fit.
#
# The doctor warns about the lag time: −2LL jumps where a dose's time plus the lag passes a
# measurement time, so gradient methods are safe only if that does not happen near the
# estimate (a profile of `tlag` would show it). And the lag time and ka have high shrinkage:
# with few early samples, each subject's own absorption is poorly determined.
#
# Next to Monolix's fit of the same model:

# %%
e = base.estimates
pairs = {"tlag.median": "Tlag_pop", "ka.median": "ka_pop", "CL.median": "Cl_pop",
         "V.median": "V_pop", "tlag.sdlog": "omega_Tlag", "ka.sdlog": "omega_ka",
         "CL.sdlog": "omega_Cl", "V.sdlog": "omega_V"}  # fmt: skip
t = base.table().set_index("quantity")
compare = pd.DataFrame({
    "tenqsp-fit": [e[q] for q in pairs], "SE": [t.loc[q, "SE"] for q in pairs],
    "Monolix": [ref_base.estimate[m] for m in pairs.values()],
    "Monolix SE": [ref_base.se[m] for m in pairs.values()]}, index=list(pairs))  # fmt: skip
# the difference against the uncertainty of both estimates
both_se = np.hypot(compare.SE, compare["Monolix SE"])
compare["difference / SE"] = (compare["tenqsp-fit"] - compare.Monolix) / both_se
print(compare.round(4).to_string())
assert (compare["difference / SE"].abs() < 1).all()

# %% [markdown]
# Every estimate agrees with Monolix's within one standard error of the difference (the two
# standard errors combined), CL and V to 1 %. They
# are not identical, for two reasons: Monolix estimates by SAEM and tenqsp-fit by FOCE-I
# (`laplace`), and Monolix's error model here adds the two parts of the SD (`a + b·f`),
# where tenqsp-fit's `combined` adds their squares. For the same reason the two −2LL are not
# comparable. The absorption, ka and its spread, differs most, and is the least determined.
#
# ## 3. Sex on the volume
#
# As Monolix: the volume of women is the volume of men times exp(`b_sex`).

# %%
SEX = "exp(b_sex) if sex == 0 else 1"
by_sex = fit(f"""estimate V_pop = 8
estimate b_sex = 0
estimate V ~ lognormal(median=V_pop*({SEX}), sdlog=0.2)
""")
row = by_sex.table().set_index("quantity").loc["b_sex"]
print(f"-2LL {base.m2ll:.1f} -> {by_sex.m2ll:.1f}; b_sex {row['estimate']:.3f} "
      f"(SE {row['SE']:.3f}); Monolix {ref_sex.estimate['beta_V_tSex_F']:.3f} "
      f"(SE {ref_sex.se['beta_V_tSex_F']:.3f})")  # fmt: skip
print(f"women's volume {np.exp(row['estimate']):.2f} times men's")
assert base.m2ll - by_sex.m2ll > 30
assert abs(row["estimate"] - ref_sex.estimate["beta_V_tSex_F"]) < row["SE"]

# %% [markdown]
# A large drop in −2LL, and the same effect as Monolix's (whose model also had sex on CL,
# where it found nothing): women have a volume about 30 % smaller. But the women here are
# also 22 kg lighter, and volume follows body size.
#
# ## 4. Weight on the volume, and both
#
# The volume in proportion to a power of weight, `(wt/70)^b_wt`; then weight and sex
# together.

# %%
by_weight = fit("""estimate V_pop = 8
estimate b_wt = 1
estimate V ~ lognormal(median=V_pop*(wt/70)^b_wt, sdlog=0.2)
""")
both = fit(f"""estimate V_pop = 8
estimate b_wt = 1
estimate b_sex = 0
estimate V ~ lognormal(median=V_pop*(wt/70)^b_wt*({SEX}), sdlog=0.2)
""")
summary = pd.DataFrame({
    "-2LL": [r.m2ll for r in (base, by_sex, by_weight, both)],
    "V.sdlog": [r.estimates["V.sdlog"] for r in (base, by_sex, by_weight, both)]},
    index=["no covariate", "sex", "weight", "weight and sex"])  # fmt: skip
print(summary.round(3).to_string())
print(both)

# %%
b = both.table().set_index("quantity")
lo, hi = b.loc["b_sex", "CI 2.5 %"], b.loc["b_sex", "CI 97.5 %"]
print(f"with weight in, b_sex {b.loc['b_sex', 'estimate']:.3f}, 95 % CI ({lo:.3f}, {hi:.3f}); "
      f"-2LL lower by {by_weight.m2ll - both.m2ll:.2f} than weight alone")  # fmt: skip
assert by_weight.m2ll < by_sex.m2ll - 5
assert by_weight.m2ll - both.m2ll < 3.84 and lo < 0 < hi

# %% [markdown]
# Weight explains the volume better than sex does (−2LL 12 lower, for the same number of
# quantities), and leaves less unexplained between subjects (the spread of V falls to
# 0.14). With weight in the model, sex adds nothing: −2LL falls by less than 1, short of the
# 3.84 a real effect would need, and its interval spans 0. The "sex effect" was the weight
# difference between these men and women. The exponent of weight is close to 1: volume in
# proportion to body weight, as expected for warfarin.
#
# Five women are too few to say that sex has no effect of its own: the interval allows a
# volume up to 25 % smaller or 11 % larger. What the data do say is that weight, not sex,
# is the covariate to put in the model.
#
# ## What this shows
#
# - The PK model agrees with Monolix's fit within the standard errors; the remaining
#   differences come from the method and the error model, not from the data.
# - A covariate that improves the fit may stand in for another: here sex for weight. Test
#   them together before reading the effect as biological.
# - A lag time makes −2LL jump; the doctor says so, and a profile of it is the check.
