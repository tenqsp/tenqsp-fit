# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""The QSP quickstart, start to end; writes the figures of the page to docs/images.
Run from docs/examples: python quickstart_qsp.py (about 15 minutes)"""

import matplotlib

matplotlib.use("Agg")
import tenqsp_fit as tf  # noqa: E402

problem = tf.load("infection_age.tq")
print(problem.summary())
result = tf.fit(problem)
print(result)
result.plot_individual().savefig("../images/timekill_fit.png", dpi=90, bbox_inches="tight")
ax = result.plot_waterfall()
ax.figure.savefig("../images/timekill_waterfall.png", dpi=90, bbox_inches="tight")

bacteria = tf.fit(tf.load("infection_age_bacteria.tq"))
print(bacteria)
bacteria.plot_trace().savefig("../images/timekill_bacteria_trace.png", dpi=90,
                              bbox_inches="tight")  # fmt: skip
