# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
#
# Reference fit: phenobarbital in 59 newborns (R's nlme::Phenobarb; data from Grasela and
# Donn, Dev Pharmacol Ther 1985), the model of NONMEM's phenobarbital example: one
# compartment, intravenous doses, CL = CL_kg Wt, V = V_kg Wt (1 + apgar if Apgar < 5),
# lognormal CL and V between infants (uncorrelated), proportional residual error. Fitted by
# R's nlme. Writes the reference results as CSV next to this file:
#
#   Rscript validation/phenobarbital/reference.R

suppressPackageStartupMessages(library(nlme))
here <- dirname(normalizePath(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))))

d <- as.data.frame(Phenobarb)
d$Subject <- as.integer(as.character(d$Subject))
doses <- d[!is.na(d$dose), c("Subject", "time", "dose")]
obs <- d[!is.na(d$conc), ]
obs$low <- as.numeric(as.integer(as.character(obs$Apgar)) < 5)
obs <- obs[, c("Subject", "time", "conc", "Wt", "low")]

# superposition of the boluses given up to each sample
pred <- function(lCl, lV, apgar, Subject, time, Wt, low) {
  out <- numeric(length(time))
  for (i in seq_along(time)) {
    dd <- doses[doses$Subject == Subject[i] & doses$time <= time[i], ]
    CL <- exp(lCl[i]) * Wt[i]
    V <- exp(lV[i]) * Wt[i] * (1 + apgar[i] * low[i])
    out[i] <- sum(dd$dose / V * exp(-CL / V * (time[i] - dd$time)))
  }
  out
}

g <- groupedData(conc ~ time | Subject, data = obs)
fit <- nlme(conc ~ pred(lCl, lV, apgar, Subject, time, Wt, low), data = g,
            fixed = lCl + lV + apgar ~ 1, random = pdDiag(lCl + lV ~ 1),
            start = c(lCl = log(0.005), lV = log(1), apgar = 0.1),
            weights = varPower(form = ~ fitted(.), fixed = 1),
            control = nlmeControl(msMaxIter = 200, pnlsTol = 1e-6, tolerance = 1e-8))

fe <- fixef(fit)
sd <- as.numeric(VarCorr(fit)[, "StdDev"])
out <- data.frame(
  quantity = c("CL_kg", "V_kg", "apgar", "CL.sdlog", "V.sdlog", "conc.proportional"),
  estimate = c(exp(fe[["lCl"]]), exp(fe[["lV"]]), fe[["apgar"]], sd[1], sd[2], sd[3])
)
write.csv(out, file.path(here, "reference_population_nlme.csv"), row.names = FALSE)
writeLines(format(-2 * as.numeric(logLik(fit)), digits = 10),
           file.path(here, "reference_population_nlme_m2ll.txt"))
print(out)
