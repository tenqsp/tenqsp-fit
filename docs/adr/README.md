# Decision records

One file per decision, `NNNN-short-title.md`, with: Context, Options considered, Measurements
(where there are any), Decision, Consequences. A record is not rewritten after it is accepted;
a later record supersedes an earlier one.

| Record | Decision | Status |
|---|---|---|
| [0001](0001-optimize-defaults.md) | `method optimize`: Levenberg–Marquardt or BFGS, multi-start, scales | accepted |
| [0002](0002-laplace-defaults.md) | `method laplace`: FOCE-I with the full outer gradient | accepted |
| [0003](0003-nuts-defaults.md) | `method nuts`: a mass matrix from the fast fit | accepted |
| [0004](0004-profile-likelihood.md) | Profiles: small-sample cut-off, scan past the walk | accepted |
