# %% [markdown]
# # Tutorial 1: theophylline, one subject, then twelve
#
# Theophylline, an oral bronchodilator, in 12 subjects (R's `Theoph` data): one dose each,
# then 11 concentrations over 24 hours. It is the textbook example of a population PK fit,
# fitted in every pharmacometrics course, so the answers are known. This tutorial fits it in
# the order a modeller learns it, and checks each answer against R's own fits (`nls` for a
# subject alone, `nlme` for the population):
#
# 1. one subject, by maximum likelihood (`method optimize`);
# 2. the same subject by Bayesian sampling (`method nuts`), next to it;
# 3. each subject alone, twelve fits;
# 4. all twelve together, with ka and CL varying between subjects (`method laplace`), and
#    what that changes for the subjects the data say least about.
#
# Every section ends with checks against R's numbers.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tenqsp as tq

import tenqsp_fit as tf

data = pd.read_csv("../examples/theoph.csv")
# R's own fits of the same data, stored in the repository (log values and their SEs)
REF = "../../validation/theophylline/"
ref_alone = pd.read_csv(REF + "reference_individual_nls.csv")
ref_pop = pd.read_csv(REF + "reference_population_nlme.csv").set_index("quantity")
print(data.head())

fig, ax = plt.subplots(figsize=(6, 3.5))
for _, d in data.groupby("Subject"):
    ax.plot(d.Time, d.conc, "o-", ms=3, lw=0.8)
ax.set_xlabel("time (h)")
ax.set_ylabel("concentration (mg/L)")
ax.set_title("theophylline, 12 subjects")

# %% [markdown]
# One row per measurement; the dose (mg/kg) is a column. The model is one compartment with
# first-order absorption, written with ka, CL and the elimination rate ke (so the volume is
# CL/ke), the parameterisation of R's `SSfol`:

# %%
model = tq.Model.from_text("""
model theoph
param ka=1.6, CL=0.04, ke=0.086
d_t(depot) = -ka*depot
d_t(centr) = ka*depot - ke*centr
observable conc = centr*ke/CL
""")

# %% [markdown]
# ## 1. One subject
#
# Subject 1 alone. The fit is built in Python here, one call per line of fit text
# (`p.to_text()` prints it): which rows are data, where the dose goes, which observable the
# measurements are, what is estimated, and the residual error.

# %%
def one_subject(subject):
    p = tf.Problem(model, name=f"subject_{subject}")
    p.data(data[data.Subject == subject], ID="Subject", TIME="Time", DV="conc")
    p.doses(column="Dose", at=0, into="depot")
    p.observe("conc", "DV")
    p.estimate("ka", 1.5)
    p.estimate("CL", 0.04)
    p.estimate("ke", 0.08)
    p.error("conc", tf.additive(1))
    return p


p1 = one_subject(1)
print(p1.to_text())

# %%
r1 = tf.fit(p1, method="optimize", progress="none")
print(r1)

# %% [markdown]
# The printout ends with the doctor: no ridge (every quantity determined within a factor 10),
# one start (other optima were not looked for), and two numerical checks, that a tighter
# solver tolerance changes nothing that matters and that the exact gradients agree with
# finite differences.
#
# R's `nls` gives the same estimates. The standard errors differ in two known ways:
#
# - `nls` is least squares and estimates the residual variance as RSS/(n − p); tenqsp-fit
#   maximises the likelihood, RSS/n. That makes tenqsp-fit's SEs smaller by
#   √((n − p)/n) = √(8/11) = 0.85, which is what CL and ke show.
# - `nls` approximates the curvature by the Gauss–Newton product of first derivatives;
#   tenqsp-fit uses the curvature of −2LL itself. They differ most for the parameter the
#   model is most nonlinear in, here ka (0.75).

# %%
r1_ref = ref_alone[ref_alone.Subject == 1].set_index("quantity")
se1 = r1.table().set_index("quantity")["SE"]
for q in ["ka", "CL", "ke"]:
    ref = np.exp(r1_ref.loc[f"log {q}", "estimate"])
    ratio = (se1[q] / r1.estimates[q]) / r1_ref.loc[f"log {q}", "se_log"]
    print(f"{q:3s} tenqsp-fit {r1.estimates[q]:.5g}   nls {ref:.5g}   SE ratio {ratio:.3f}")
    assert abs(r1.estimates[q] / ref - 1) < 1e-3
    assert 0.7 < ratio < 0.9

# %% [markdown]
# ## 2. The same subject, by sampling
#
# `method nuts` draws from the posterior distribution: the likelihood times a prior for each
# quantity. None is written here, so each gets the default, a wide lognormal centred on its
# start (95 % within a factor 50). The defaults: 4 chains, each of 1000 warmup steps and 1000
# draws, each on its own core.

# %%
p1n = one_subject(1)
p1n.method("nuts")
p1n.options(seed=1)
r1n = tf.fit(p1n, progress="none")
print(r1n)

# %% [markdown]
# The data, not the wide priors, decide: the posterior medians of ka, CL and ke sit on the
# maximum-likelihood estimates. The 95 % intervals are wider, by about a third. Section 1's
# intervals come from the curvature at the estimates, which assumes many measurements; with
# 11 measurements and 4 quantities the posterior also carries how uncertain the residual SD
# is. The maximum-likelihood residual SD (0.62) is the one that is biased low with few
# measurements, by √((n − p)/n) as in section 1, and the posterior median is higher (0.77).
#
# Sampling earns its cost in exactly such cases: few data, a weakly determined quantity, a
# skewed or bounded one, or prior knowledge that should count.

# %%
ml, post = r1.table().set_index("quantity"), r1n.table().set_index("quantity")
print(pd.DataFrame({"optimize": ml.estimate, "nuts median": post["median"],
                    "optimize 95 %": ml["CI 2.5 %"].round(4).astype(str) + " to "
                    + ml["CI 97.5 %"].round(4).astype(str),
                    "nuts 95 %": post["CI 2.5 %"].round(4).astype(str) + " to "
                    + post["CI 97.5 %"].round(4).astype(str)}))  # fmt: skip
for q in ["ka", "CL", "ke"]:
    assert abs(np.log(post["median"][q] / ml.estimate[q])) < 0.25 * se1[q] / ml.estimate[q]
for q in ml.index:  # wider intervals, every quantity
    assert post["CI 97.5 %"][q] - post["CI 2.5 %"][q] > ml["CI 97.5 %"][q] - ml["CI 2.5 %"][q]
assert r1n.doctor().count("warning") == 0

# %% [markdown]
# ## 3. Each subject alone
#
# Twelve fits, one per subject, as R's `nlsList` does.

# %%
alone = {}
for subject in sorted(data.Subject.unique()):
    r = tf.fit(one_subject(subject), method="optimize", progress="none")
    assert r.status == "converged"
    alone[subject] = r
table_alone = pd.DataFrame(
    {q: [alone[s].estimates[q] for s in alone] for q in ["ka", "CL", "ke"]}, index=list(alone)
)
table_alone["RSE % of ka"] = [
    alone[s].table().set_index("quantity").loc["ka", "RSE %"] for s in alone
]
print(table_alone.round(4))

# every subject's estimates are R's
for q in ["ka", "CL", "ke"]:
    ref = np.exp(ref_alone[ref_alone.quantity == f"log {q}"].set_index("Subject").estimate)
    assert np.max(np.abs(table_alone[q] / ref - 1)) < 1e-3

# %% [markdown]
# CL and ke are well determined for every subject. ka is not: it is set by the first hour,
# where each subject has three or four samples, and for subject 9 it is 8.9 per hour with a
# large uncertainty. A subject alone says nothing about what is typical; the next section
# lets the subjects inform each other.

# %% [markdown]
# ## 4. All twelve together
#
# Now ka and CL vary between subjects, lognormally around a typical value (the median), with
# a spread (sdlog, about the CV for small values); ke is one value for everyone, as in R's
# `nlme` fit of these data. `method laplace` is first-order conditional estimation with
# interaction (FOCE-I), here written in the fit text:

# %%
pop = tf.Problem.from_text(
    """
fit theoph_pop
data "../examples/theoph.csv" with ID=Subject, TIME=Time, DV=conc
unit ID
doses Dose at t=0 into depot
observe conc = DV

estimate ka ~ lognormal(median=1.5, sdlog=0.5)
estimate CL ~ lognormal(median=0.04, sdlog=0.5)
estimate ke = 0.08
error conc ~ additive(1)

method laplace
""",
    model=model,
)
rp = tf.fit(pop, progress="none")
print(rp)

# %% [markdown]
# R's `nlme` reaches −2LL 354.04; tenqsp-fit's 353.98 is a little lower. The two methods
# approximate the same likelihood in different ways (`nlme` by Lindstrom–Bates steps,
# tenqsp-fit by the Laplace approximation), so they land close but not on the same point.
# Every estimate agrees with `nlme`'s within its standard error:

# %%
names = {"ka.median": "log ka.median", "CL.median": "log CL.median", "ke": "log ke",
         "ka.sdlog": "ka.sdlog", "CL.sdlog": "CL.sdlog",
         "conc.additive": "conc.additive"}  # fmt: skip
rows = []
for ours, theirs in names.items():
    value = ref_pop.loc[theirs, "estimate"]
    value = np.exp(value) if theirs.startswith("log ") else value
    rows.append((ours, rp.estimates[ours], value, ref_pop.loc[theirs, "se_log"]))
compare = pd.DataFrame(rows, columns=["quantity", "tenqsp-fit", "nlme", "nlme SE (log)"])
compare["log difference"] = np.log(compare["tenqsp-fit"] / compare["nlme"])
print(compare.round(5))
assert rp.m2ll < 354.0428
assert np.all(np.abs(compare["log difference"]) <= compare["nlme SE (log)"])

# %% [markdown]
# ### What the population changes for one subject
#
# `rp.individual()` gives each subject's own ka and CL in the population fit (the conditional
# modes: NONMEM's empirical Bayes estimates). Two things change from the fits of section 3:
#
# - **A subject whose data say little about ka borrows from the others.** Subject 9's ka, the
#   least determined (RSE 32 %), moves from 8.9 to 6.6 per hour, toward the typical value;
#   it moves furthest of all.
# - **ke is now one value for everyone** (0.085 per hour), so a subject whose own ke was
#   different refits its ka and CL around it: subject 1's ke alone was 0.054, and its ka
#   moves from 1.78 to 1.42.
# %%
ind = rp.individual().set_index("unit")
both = pd.DataFrame({"ka alone": table_alone["ka"], "ka in the population": ind["ka"],
                     "RSE % alone": table_alone["RSE % of ka"]})  # fmt: skip
print(both.round(3))

fig, ax = plt.subplots(figsize=(5, 3.5))
ax.scatter(both["ka alone"], both["ka in the population"])
for s, row in both.iterrows():
    ax.annotate(str(s), (row["ka alone"], row["ka in the population"]), fontsize=8)
lim = [0.5, 10]
ax.plot(lim, lim, "k:", lw=0.8, label="no change")
ax.axhline(rp.estimates["ka.median"], color="grey", lw=0.8, label="typical ka")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("ka, subject alone (1/h)")
ax.set_ylabel("ka in the population fit (1/h)")
ax.legend(fontsize="small")

typical = rp.estimates["ka.median"]
pull = np.abs(np.log(both["ka alone"] / typical)) - np.abs(
    np.log(both["ka in the population"] / typical)
)
assert pull[9] == pull.max() and pull[9] > 0.25  # subject 9 moves most, toward the typical ka
assert both["ka in the population"].idxmax() == 9  # and still has the fastest absorption

# %% [markdown]
# The population fit's shrinkage (`rp.shrinkage`) is small for both ka and CL, so the
# subjects' values still mostly reflect their own data; with sparse sampling it grows, and
# the doctor warns above 30 %.
#
# For NONMEM users, `rp.tables()` gives the sdtab and patab tables:

# %%
tables = rp.tables()
sdtab, patab = tables["sdtab"], tables["patab"]
print(sdtab.head())
print(patab.head())
print(rp.shrinkage)
assert all(abs(v) < 0.1 for v in rp.shrinkage.values())
assert set(["ID", "TIME", "DV", "PRED", "IPRED", "IWRES"]) <= set(sdtab.columns)

# %% [markdown]
# ## What this shows
#
# - The same model fitted three ways gives R's answers: `optimize` reproduces `nls` for every
#   subject, and `laplace` agrees with `nlme` within the standard errors, at a −2LL as low.
# - `nuts` puts ka, CL and ke where `optimize` does, with wider intervals that account for
#   having only 11 measurements; it costs about two minutes for one subject.
# - A population fit keeps what each subject's data determine and fills in what they do not:
#   subject 9's ka moves from 8.9 toward the typical value.
#
# The [population quickstart](../start/quickstart-population.md) continues from this fit:
# goodness of fit, a visual predictive check, a profile of ke, and a prediction for a new
# regimen.
