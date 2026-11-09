# %% [markdown]
# # Tutorial 2: a transit delay, then absorption: what do the data determine?
#
# A drug that passes a transit delay before it is absorbed is described by two quantities:
# the mean transit time (MTT) and the absorption rate constant (ka). Both slow the rise of
# the concentration, so the question every modeller meets is whether the data can tell
# them apart. This tutorial answers it for simulated studies, where the true values are
# known:
#
# 1. the model of TenQSP's first tutorial, and studies simulated from it;
# 2. a sparse design (samples from 2 h on): the doctor finds MTT and ka moving together, and
#    a profile shows what the data do determine, the mean absorption time MTT + 1/ka;
# 3. a rich design (samples from 15 min on): both are estimated, but the interval for ka
#    from its standard error is not the one to trust;
# 4. ten studies of the rich design: the estimates of ka scatter, and the profile
#    intervals contain the true value where the standard-error intervals sometimes do not.

# %%
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tenqsp as tq

import tenqsp_fit as tf

TRUE = {"MTT": 1.5, "ka": 3.0, "CL": 3.0, "V": 30.0}

# %% [markdown]
# ## 1. The model and the studies
#
# As in [TenQSP's tutorial 1](https://tenqsp.github.io/tenqsp/tutorials/01_transit_compartments/):
# the drug in transit is a field `u(z)` along an axis from 0 (just dosed) to 1 (reaching the
# absorption compartment `Aa`), crossed in MTT on average. Four cells of the `erlang` scheme
# are exactly three transit compartments. The dose D starts in the first cell.

# %%
model = tq.Model.from_text("""
model transit
axis z in [0, 1] n=4 kind=one_way
param MTT=1.5, ka=3, CL=3, V=30, D=100
field u(z)
d_t(u) = -d_z(u/MTT)
    at z=1: outflow
d_t(Aa) = u(z=1)/MTT - ka*Aa
d_t(Ac) = ka*Aa - CL/V*Ac
u(0) = D*point(z, at=0.125)
observable conc = Ac/V
""")

SPARSE = [2, 4, 8, 24]
RICH = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 3, 4, 8, 24]


def study(times, seed, subjects=12, cv=0.15):
    """Concentrations of `subjects` subjects at `times`, from the true values, with a
    measurement error of `cv` (lognormal)."""
    true = np.asarray(model.solve((0, 24), times=times, rtol=1e-10, atol=1e-12)["conc"])
    rng = np.random.default_rng(seed)
    rows = [
        {"ID": i, "TIME": t, "DV": c * np.exp(cv * rng.standard_normal())}
        for i in range(1, subjects + 1)
        for t, c in zip(times, true, strict=True)
    ]
    return pd.DataFrame(rows)


t = np.linspace(0, 24, 241)
fine = np.asarray(model.solve((0, 24), times=t)["conc"])
fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
for a, times, name in [(ax[0], SPARSE, "sparse"), (ax[1], RICH, "rich")]:
    d = study(times, seed=2)
    a.plot(d.TIME, d.DV, "o", ms=3, alpha=0.5)
    a.plot(t, fine, "k-", lw=1, label="true")
    a.set_xlim(0, 12)
    a.set_title(f"{name} design: {len(times)} samples per subject")
    a.set_xlabel("time (h)")
    a.set_ylabel("concentration")
    a.legend()

# %% [markdown]
# Every subject has the same parameters here, to keep the question about the design;
# parameters that vary between subjects come in the next tutorials.
#
# The fit estimates MTT, ka, CL and V, and a proportional error. ka gets a lower bound,
# 0.3 per hour, three times the elimination rate CL/V: with oral data alone, a slow
# absorption and a fast elimination fit as well as the reverse (flip-flop kinetics), and only
# an intravenous dose could tell them apart. The bound says which we mean.


# %%
def problem(data, checks="full"):
    p = tf.Problem(model, name="transit")
    p.data(data)
    p.unit("ID")
    p.observe("conc", "DV")
    p.estimate("MTT", 1.0)
    p.estimate("ka", 2.0, bounds=(0.3, 100))
    p.estimate("CL", 2.0)
    p.estimate("V", 20.0)
    p.fix("D", 100)
    p.error("conc", tf.proportional(0.2))
    p.method("optimize")
    p.options(starts=4, seed=1, checks=checks, progress="none")
    return p


# %% [markdown]
# ## 2. The sparse design
#
# Four samples per subject, the first at 2 h, when the concentration is near its peak.

# %%
r_sparse = tf.fit(problem(study(SPARSE, seed=2)))
print(r_sparse)

# %% [markdown]
# The estimates of CL and V are fine (2-4 % RSE). The doctor warns that MTT and ka are weakly
# determined and move together (a correlation of 1.00): the data are fitted almost equally
# well by a long delay with fast absorption as by a short delay with slower absorption.
#
# A profile of ka shows what that means. Each point holds ka at one value and refits the
# rest; the 95 % interval is where −2LL stays within the cut-off of its minimum. Here ka can
# be anywhere from about 0.6 to the upper bound: the upper end is not reached.

# %%
prof = r_sparse.profile("ka")
print(prof)
prof.plot()

# %% [markdown]
# What stays fixed along the profile is what the data determine. MTT + 1/ka is the mean
# time from the dose until the drug reaches the circulation (the mean absorption time,
# MAT): the mean time in transit plus the mean time in `Aa`. Within the profile's interval,
# ka goes from 0.66 to the bound while MAT moves only from 2.1 to 1.87 h, and above the
# estimate not at all; it is close to the true 1.5 + 1/3 = 1.83 h.

# %%
d = prof.table.copy()
d["MAT"] = d.MTT + 1 / d.value
inside = d[d["-2LL"] <= prof.m2ll + prof.cut]
print(inside[["value", "-2LL", "MTT", "MAT"]].round(3).to_string(index=False))
print(f"MAT within the interval: {inside.MAT.min():.3f} to {inside.MAT.max():.3f} h")

assert r_sparse.doctor().count("warning") >= 1
assert "weakly determined" in str(r_sparse.doctor())
assert np.isnan(prof.interval[1])  # the upper end is not reached
assert inside.MAT.max() / inside.MAT.min() < 1.15  # MAT moves by less than 15 %
assert abs(inside.MAT.median() / (TRUE["MTT"] + 1 / TRUE["ka"]) - 1) < 0.05

# %% [markdown]
# ## 3. The rich design
#
# Eleven samples per subject, from 15 minutes on, so the rise is sampled.

# %%
r_rich = tf.fit(problem(study(RICH, seed=100)))
print(r_rich)
prof_rich = r_rich.profile("ka")
print(prof_rich)
prof_rich.plot()

# %% [markdown]
# Both MTT and ka are now estimated, and the doctor no longer finds them weakly determined.
# (Its other warning, several optima, is one of the four starts stopping at a much worse
# fit, −2LL 135 higher; three reached the best.)
#
# There are two intervals for ka. The one in the table comes from the standard error: it
# assumes that −2LL is a parabola around the estimate (on the log scale of ka). The profile
# follows −2LL itself. For this study they differ; which to trust is the subject of the
# next section.
# %%
row = r_rich.table().set_index("quantity").loc["ka"]
print(f"ka {r_rich.estimates['ka']:.3g}: from its SE ({row['CI 2.5 %']:.3g}, "
      f"{row['CI 97.5 %']:.3g}); from the profile ({prof_rich.interval[0]:.3g}, "
      f"{prof_rich.interval[1]:.3g}); true value {TRUE['ka']}")  # fmt: skip
assert "weakly determined" not in str(r_rich.doctor())
assert prof_rich.interval[0] <= TRUE["ka"] <= prof_rich.interval[1]

# %% [markdown]
# ## 4. Ten studies
#
# Is the profile's interval the one to trust? Simulate ten studies of the rich design, fit
# each, and count how often each interval contains the true ka. (A check of the 95 %
# intervals would need hundreds of studies; ten show the pattern. The quick checks of the
# doctor are enough here.)

# %%
rows = []
with warnings.catch_warnings():
    warnings.simplefilter("ignore")  # warnings of weakly determined fits, for some studies
    for seed in range(100, 110):
        r = tf.fit(problem(study(RICH, seed), checks="quick"))
        ci = r.table().set_index("quantity").loc["ka"]
        pr = r.profile("ka")
        rows.append({"study": seed - 99, "ka": r.estimates["ka"],
                     "SE interval": (round(ci["CI 2.5 %"], 3), round(ci["CI 97.5 %"], 3)),
                     "profile interval": (round(pr.interval[0], 3), round(pr.interval[1], 3)),
                     "SE covers": ci["CI 2.5 %"] <= TRUE["ka"] <= ci["CI 97.5 %"],
                     "profile covers": pr.interval[0] <= TRUE["ka"] <= pr.interval[1]})  # fmt: skip
studies = pd.DataFrame(rows)
print(studies.to_string(index=False))
print(f"true ka {TRUE['ka']}: inside the SE interval in {studies['SE covers'].sum()} of 10, "
      f"inside the profile interval in {studies['profile covers'].sum()} of 10")  # fmt: skip

fig, ax = plt.subplots(figsize=(6, 3.5))
for i, s in studies.iterrows():
    ax.plot(s["SE interval"], [i - 0.12] * 2, color="#888888", lw=2)
    ax.plot(s["profile interval"], [i + 0.12] * 2, color="#1f3b73", lw=2)
    ax.plot(s["ka"], i, "k.", ms=6)
ax.plot([], [], color="#888888", lw=2, label="from the SE")
ax.plot([], [], color="#1f3b73", lw=2, label="profile")
ax.axvline(TRUE["ka"], color="#d62728", ls="--", lw=1, label="true ka")
ax.set_xscale("log")
ax.set_xlabel("ka (1/h)")
ax.set_ylabel("study")
ax.legend(fontsize="small")

assert studies["profile covers"].all()
assert studies.ka.max() / studies.ka.min() > 1.5  # the estimates scatter

# %% [markdown]
# The estimates of ka scatter between studies of the same design, from 1.4 to 5.4 per hour.
# The intervals from the standard error are sometimes far too wide (study 3: 0.5 to 26) and
# sometimes too narrow and in the wrong place, missing the true value (studies 6 and 9). The
# profile intervals contain it in every study.
#
# ## What this shows
#
# - With samples only around and after the peak, the data determine the mean absorption
#   time MTT + 1/ka, not MTT and ka apart. The doctor says so (weakly determined, moving
#   together), and the profile shows the trade-off.
# - Sampling the rise determines both, but ka's likelihood is far from a parabola: its
#   interval should come from a profile (`result.profile("ka")`), not from its standard error.
# - Oral data alone cannot tell absorption from elimination (flip-flop); a bound on ka says
#   which one is meant.
