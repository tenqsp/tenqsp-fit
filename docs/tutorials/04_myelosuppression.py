# %% [markdown]
# # Tutorial 4: myelosuppression, from a dose-ranging study to a dose
#
# A cytotoxic drug lowers the neutrophil count for about two weeks after each dose; too low a
# count (below 0.5 × 10⁹/L, grade 4 neutropenia) puts the patient at risk of infection. The
# model of Friberg et al. (2002) describes it: cells proliferate, mature through a chain of
# transit stages, and circulate; the drug kills proliferating cells, and a low count speeds
# up proliferation (the feedback, with exponent γ).
#
# This tutorial plays out the use the model was made for, on a simulated study whose true
# values are known:
#
# 1. a dose-ranging study: 20 patients at 60, 90 or 120 mg, neutrophils counted 8 times over
#    3 weeks;
# 2. the population fit: are the true values recovered?
# 3. a prediction for a dose not studied;
# 4. the dose: the highest dose at which at most 10 % of patients reach grade 4.

# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tenqsp as tq

import tenqsp_fit as tf

TRUE = {"MTT": 100.0, "gamma": 0.17, "circ0": 5.0, "slope": 0.5}
OMEGA = 0.3  # between patients: lognormal, sdlog 0.3, on circ0 and on slope

# %% [markdown]
# ## 1. The model and the study
#
# The model is [TenQSP's tutorial 3](https://tenqsp.github.io/tenqsp/tutorials/03_myelosuppression/):
# maturation is an axis `z` from 0 to 1, cut into three cells (three transit compartments).
# Time is in hours, the drug amount in mg, the count in 10⁹/L.

# %%
model = tq.Model.from_text("""
model friberg
axis z in [0, 1] n=3 kind=one_way
param MTT=100, gamma=0.17, circ0=5, slope=0.5, CL=1, V=50
ktr = 4/MTT
T_mat = 3/ktr
Conc = Ac/V
E_drug = slope*Conc
field Transit(z)
d_t(Prol) = ktr*Prol*(1 - E_drug)*(circ0/Circ)^gamma - ktr*Prol
d_t(Transit) = -d_z(Transit/T_mat)
    at z=0: Transit/T_mat = ktr*Prol
    at z=1: outflow
d_t(Circ) = Transit(z=1)/T_mat - ktr*Circ
d_t(Ac) = -CL/V*Ac
Prol(0) = circ0
Transit(0) = ktr*circ0*T_mat
Circ(0) = circ0
observable ANC = Circ
""")

DAYS = np.array([0, 3, 7, 9, 11, 14, 18, 21])
rng = np.random.default_rng(1)
rows = []
for i in range(1, 21):
    dose = [60, 90, 120][i % 3]
    own = {k: TRUE[k] * np.exp(OMEGA * rng.standard_normal()) for k in ("circ0", "slope")}
    sol = model.solve((0, 504), times=24 * DAYS, params=own,
                      regimen=tq.bolus("Ac", dose, times=[0]), rtol=1e-8, atol=1e-10)  # fmt: skip
    anc = np.asarray(sol["ANC"]) * np.exp(0.15 * rng.standard_normal(len(DAYS)))
    rows.append(dict(ID=i, TIME=0.0, AMT=dose, EVID=1, DV=np.nan))
    rows += [dict(ID=i, TIME=24.0 * d, AMT=np.nan, EVID=0, DV=v)
             for d, v in zip(DAYS, anc, strict=True)]  # fmt: skip
data = pd.DataFrame(rows)
dose_of = data[data.EVID == 1].set_index("ID").AMT

fig, ax = plt.subplots(figsize=(6, 3.5))
colors = {60: "#9ecae1", 90: "#4292c6", 120: "#08306b"}
for i, d in data[data.EVID == 0].groupby("ID"):
    ax.plot(d.TIME / 24, d.DV, "o-", ms=3, lw=0.8, color=colors[dose_of[i]])
for dose, c in colors.items():
    ax.plot([], [], color=c, label=f"{dose} mg")
ax.axhline(0.5, color="#d62728", ls="--", lw=1, label="grade 4")
ax.set_yscale("log")
ax.set_xlabel("day")
ax.set_ylabel("neutrophils (10⁹/L)")
ax.legend(fontsize="small")

# %% [markdown]
# The count falls to its lowest (the nadir) 9 to 11 days after the dose and recovers by
# day 21, deeper at higher doses. The baseline count (`circ0`) and the drug's potency
# (`slope`) vary between patients; the measurement error is 15 %.
#
# ## 2. The population fit
#
# `circ0` and `slope` vary between patients (lognormal); MTT and γ are one value each. The
# starts are deliberately away from the truth.

# %%
p = tf.Problem(model, name="friberg")
p.data(data)
p.unit("ID")
p.doses(into="Ac")
p.observe("ANC", "DV")
p.estimate("circ0", tf.lognormal(median=4, sdlog=0.5))
p.estimate("slope", tf.lognormal(median=0.3, sdlog=0.5))
p.estimate("MTT", 80)
p.estimate("gamma", 0.25)
p.error("ANC", tf.proportional(0.2))
p.method("laplace")
p.options(progress="none")
r = tf.fit(p)
print(r)

# %%
t = r.table().set_index("quantity")
truth = {"circ0.median": TRUE["circ0"], "slope.median": TRUE["slope"], "MTT": TRUE["MTT"],
         "gamma": TRUE["gamma"], "circ0.sdlog": OMEGA, "slope.sdlog": OMEGA,
         "ANC.proportional": 0.15}  # fmt: skip
ci = t.loc[list(truth), ["CI 2.5 %", "CI 97.5 %"]].to_numpy()
check = pd.DataFrame({"true": list(truth.values()),
                      "estimate": t.loc[list(truth), "estimate"].to_numpy(),
                      "95 % CI": [f"({a:.3g}, {b:.3g})" for a, b in ci]},
                     index=list(truth))  # fmt: skip
print(check.to_string())
inside = [(t.loc[q, "CI 2.5 %"] <= v <= t.loc[q, "CI 97.5 %"]) for q, v in truth.items()]
assert sum(inside) >= len(truth) - 1  # one miss in seven is chance at 95 %

# %% [markdown]
# Every true value is recovered, within its interval: twenty patients with eight counts each
# determine the maturation time and the feedback well (MTT to 2 %, γ to 5 %). The shrinkage
# is low, so each patient's own `circ0` and `slope` are well determined too.
#
# ## 3. A dose not studied
#
# The study stopped at 120 mg. What would 150 mg do? `result.predict` runs the fitted model
# for a new regimen: the typical patient (with the uncertainty of the estimates), and with
# `between_units=True` also the range of a new patient.

# %%
pred = r.predict(regimen=tq.bolus("Ac", 150, times=[0]), times=np.linspace(0, 504, 85),
                 between_units=True)  # fmt: skip
print(pred.summary())
fig = pred.plot(log=True)
fig.axes[0].axhline(0.5, color="#d62728", ls="--", lw=1)
nadir = pred.table.loc[pred.table["median"].idxmin()]
lo, hi = nadir["new unit 5 %"], nadir["new unit 95 %"]
print(f"typical nadir {nadir['median']:.2f} on day {nadir.time / 24:.1f}; "
      f"a new patient's 90 % range at that time {lo:.2f} to {hi:.2f}")  # fmt: skip
assert nadir["new unit 5 %"] < 0.5 < nadir["median"]

# %% [markdown]
# The typical patient stays above the grade 4 line at 150 mg, but the lower end of a new
# patient's range goes well below it. How many patients would that be?
#
# ## 4. Choosing the dose
#
# A virtual population: a thousand patients, each with its own `circ0` and `slope` drawn
# from the fitted distributions, run through the fitted model (`r.model()`, the model with
# the estimates in) at each dose. TenQSP compiles the simulation once as a JAX function of
# `circ0` and `slope`, and `jax.vmap` runs all thousand patients in one call. The same
# patients through the true model show what the answer should be.


# %%
def grade4(m, circ0, slope, sdlog_c, sdlog_s, doses, n=1000, seed=3):
    """The fraction of n patients whose count falls below 0.5 at each dose."""
    z = np.random.default_rng(seed).standard_normal((n, 2))
    X = jnp.asarray(np.column_stack([circ0 * np.exp(sdlog_c * z[:, 0]),
                                     slope * np.exp(sdlog_s * z[:, 1])]))  # fmt: skip
    sim = m.simulator(outputs=["ANC"])
    out = []
    for dose in doses:
        sc = tq.Scenario(times=np.linspace(0, 504, 169), regimen=tq.bolus("Ac", dose, times=[0]))
        f = sim.function(sc, wrt=["circ0", "slope"], derivatives=False)
        anc = np.asarray(jax.jit(jax.vmap(lambda x, f=f: f(x)["ANC"]))(X))
        out.append(float(np.mean(np.min(anc, axis=1) < 0.5)))
    return np.array(out)


doses = np.arange(60, 181, 10)
e = r.estimates
fitted = grade4(r.model(), e["circ0.median"], e["slope.median"], e["circ0.sdlog"],
                e["slope.sdlog"], doses)  # fmt: skip
true = grade4(model, TRUE["circ0"], TRUE["slope"], OMEGA, OMEGA, doses)
print(pd.DataFrame({"dose": doses, "fitted": fitted.round(3), "true": true.round(3)})
      .to_string(index=False))  # fmt: skip

fig, ax = plt.subplots(figsize=(5.5, 3.5))
ax.plot(doses, 100 * true, "k--", label="true model")
ax.plot(doses, 100 * fitted, "o-", color="#1f3b73", label="fitted model")
ax.axhline(10, color="#d62728", lw=1)
ax.axvspan(60, 120, color="0.9", zorder=0, label="doses studied")
ax.set_xlabel("dose (mg)")
ax.set_ylabel("patients with grade 4 (%)")
ax.legend(fontsize="small")

best_fit, best_true = doses[fitted <= 0.10].max(), doses[true <= 0.10].max()
print(f"highest dose with at most 10 % grade 4: {best_fit} mg (fitted), {best_true} mg (true)")
assert abs(best_fit - best_true) <= 10
assert np.all(np.diff(fitted) >= 0) and fitted[-1] > 0.5

# %% [markdown]
# The fitted model's answer is one step of 10 mg from the true one, and on the risky side:
# between 90 and 130 mg it puts fewer patients in grade 4 than the true model does. The
# typical patient is well estimated (section 2); what decides the fraction is the patients
# in the tail, and so the spread between patients. The study estimated it at 0.26 for both
# `circ0` and `slope`, below the true 0.3 though within its interval: twenty patients
# determine a spread only loosely. With the true spread in place of the estimated one, the
# fitted model comes within 3 points of the true fractions at every dose:

# %%
spread = grade4(r.model(), e["circ0.median"], e["slope.median"], OMEGA, OMEGA, doses)
print(pd.DataFrame({"dose": doses, "fitted": fitted.round(3), "fitted, true spread":
                    spread.round(3), "true": true.round(3)}).to_string(index=False))  # fmt: skip
assert np.abs(spread - true).max() < 0.5 * np.abs(fitted - true).max()

# %% [markdown]
# So the answer to "which dose" depends most on the quantity the study determines worst.
# A dose chosen this way should carry that uncertainty: draw the spread (and the other
# estimates) from their uncertainty as well, as `result.predict` does for one patient, and
# report the fraction with an interval; or study more patients.
#
# ## What this shows
#
# - A semi-mechanistic model, a field over a maturation axis with feedback, fits like any
#   other: `laplace` recovers its values from a 20-patient study.
# - `result.predict` answers "what would this dose do" for the typical patient and a new
#   one; `result.model()` gives the fitted model itself, for any simulation, such as a
#   thousand virtual patients run in one JAX call.
# - The decision a dose-ranging study is for, the highest dose with acceptable toxicity,
#   depends on the tail of the population, and so on the spread between patients, which a
#   small study estimates loosely: here one dose step off, on the risky side.
