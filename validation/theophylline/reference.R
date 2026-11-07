# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
#
# Reference fits: theophylline (R's datasets::Theoph, 12 subjects, oral dose; data from
# Boeckmann, Sheiner and Beal, NONMEM Users Guide Part V, 1994), the
# one-compartment model with first-order absorption, fitted by R's nlme (population) and nls
# (each subject). Writes the data and the reference results as CSV next to this file:
#
#   Rscript validation/theophylline/reference.R
#
# The model, as SSfol: conc = Dose ka ke / (CL (ka - ke)) (exp(-ke t) - exp(-ka t)), with
# lKe = log ke, lKa = log ka, lCl = log CL (V = CL / ke).

suppressPackageStartupMessages(library(nlme))
here <- dirname(normalizePath(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))))

data <- as.data.frame(Theoph)
data$Subject <- as.integer(as.character(data$Subject))
data <- data[order(data$Subject, data$Time), c("Subject", "Wt", "Dose", "Time", "conc")]
write.csv(data, file.path(here, "theoph.csv"), row.names = FALSE)

z <- qnorm(0.975)
se_log <- function(ci) (log(ci[3]) - log(ci[1])) / (2 * z)  # from a Wald interval on the log scale

# ---- population: ka and CL vary between subjects (lognormal, uncorrelated), ke one value,
# additive residual error; maximum likelihood
pop <- nlme(conc ~ SSfol(Dose, Time, lKe, lKa, lCl), data = Theoph,
            fixed = lKe + lKa + lCl ~ 1, random = pdDiag(lKa + lCl ~ 1),
            start = c(lKe = -2.5, lKa = 0.5, lCl = -3.2), method = "ML")
fe <- summary(pop)$tTable
iv <- intervals(pop, which = "var-cov")
sd_re <- iv$reStruct$Subject
sig <- iv$sigma
rows <- data.frame(
  quantity = c("log ka.median", "log CL.median", "log ke", "ka.sdlog", "CL.sdlog",
               "conc.additive"),
  estimate = c(fe["lKa", "Value"], fe["lCl", "Value"], fe["lKe", "Value"],
               sd_re["sd(lKa)", "est."], sd_re["sd(lCl)", "est."], sig[["est."]]),
  # SE on the log scale: fixed effects as nlme reports them (already logs); SDs from their
  # intervals, which nlme computes on the log scale
  se_log = c(fe["lKa", "Std.Error"], fe["lCl", "Std.Error"], fe["lKe", "Std.Error"],
             se_log(unlist(sd_re["sd(lKa)", ])), se_log(unlist(sd_re["sd(lCl)", ])),
             se_log(c(sig[["lower"]], sig[["est."]], sig[["upper"]])))
)
write.csv(rows, file.path(here, "reference_population_nlme.csv"), row.names = FALSE)
writeLines(sprintf("%.6f", -2 * as.numeric(logLik(pop))), file.path(here, "reference_population_nlme_m2ll.txt"))

# ---- each subject alone: least squares (nls), ka, CL, ke
ind <- nlsList(conc ~ SSfol(Dose, Time, lKe, lKa, lCl) | Subject, data = Theoph)
out <- do.call(rbind, lapply(names(ind), function(s) {
  f <- ind[[s]]
  t <- summary(f)$coefficients
  data.frame(Subject = as.integer(s),
             quantity = c("log ka", "log CL", "log ke"),
             estimate = t[c("lKa", "lCl", "lKe"), "Estimate"],
             se_log = t[c("lKa", "lCl", "lKe"), "Std. Error"],
             residual_sd = summary(f)$sigma, n = length(resid(f)))
}))
write.csv(out[order(out$Subject), ], file.path(here, "reference_individual_nls.csv"), row.names = FALSE)
cat(R.version.string, "nlme", as.character(packageVersion("nlme")), "\n")
print(rows)
