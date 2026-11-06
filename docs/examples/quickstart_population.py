# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""The population quickstart, start to end; writes the figures of the page to docs/images.
Run from docs/examples: python quickstart_population.py"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tenqsp as tq  # noqa: E402

import tenqsp_fit as tf  # noqa: E402

problem = tf.load("theoph_pop.tq")
print(problem.summary())

result = tf.fit(problem)
print(result)
print(result.individual().head(4))

result.plot_gof().savefig("../images/theoph_gof.png", dpi=90, bbox_inches="tight")
v = result.vpc()
v.figure.savefig("../images/theoph_vpc.png", dpi=90, bbox_inches="tight")

prof = result.profile("ke")
print(prof)
prof.plot()
plt.savefig("../images/theoph_profile_ke.png", dpi=90, bbox_inches="tight")

dose = tq.bolus("depot", 4.0, times=[0, 12, 24])  # 4 mg/kg every 12 h, three doses
pred = result.predict(regimen=dose, times=np.linspace(0, 48, 97), outputs=["conc"],
                      name="4 mg/kg q12h", n=200)  # fmt: skip
print(pred.table.head(6))
pred.plot()
plt.savefig("../images/theoph_predict.png", dpi=90, bbox_inches="tight")

print(result.passport())
result.save("theoph_run")
