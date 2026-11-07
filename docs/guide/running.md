# Running a fit

## While it runs

A fit shows its progress in one of three ways, set by `options progress=` in the fit text, or
by `tf.fit(problem, progress=...)` for one call:

| `progress` | Shows |
|---|---|
| `text` | the default outside a notebook: in a terminal, a block rewritten in place; elsewhere (a log file, an IDE's output pane), a line every 10 seconds and one at the end |
| `plot` | the default in a notebook (Jupyter, Positron, VS Code): a live plot, redrawn every 2 seconds |
| `none` | nothing |

A line of the text, from the theophylline fit:

```text
laplace  1 start: 1 running, 0 done   steps 13-13   best -2LL 353.9862   0:26  |  ka.median 1.623  ka.sdlog 0.6546  CL.median 0.03947  CL.sdlog 0.1684  ke 0.08496  conc.additive 0.7082
```

the method; how many starts are running and done; the fewest and most steps taken by any
start; the best −2LL so far and the time; after the bar, the values of the start with the best
−2LL. When the fit ends, `checking the fit: the solver tolerance and the gradients ...` marks
the doctor's checks.

**The live plot** shows −2LL against the step and one small panel per quantity, one line per
start, the best dark, as Monolix does. In a notebook it is redrawn in place. Outside a notebook
it is a page that refreshes itself, opened in the browser; its address is printed
(`live plot: file:///...`). `result.plot_trace()` draws the same figure after the fit.

**Ctrl-C** (or a notebook's stop button) stops the fit and keeps what it has: each start
finishes the step it is taking, and the result is marked `stopped`, at the best values so far,
without standard errors (the curvature away from a minimum means nothing). The doctor's checks
can be stopped the same way.

**A time limit** does the same when the time is up: `options timeout=2h` (seconds, or text
such as `90s`, `30min`, `2h`; off by default). The theophylline fit with `timeout=20s`:

```text
theoph_pop: method laplace (BFGS, FOCE-I), stopped; -2LL 359.8725 (132 measurements, 12 units)
starts: 1; 1 at the best fit (within 0.1); redrawn 0; failed runs during the fit 0; 28 s

quantity                  estimate         SE  RSE %   95 % CI                 also
ka.median                   1.3217
ka.sdlog                   0.48643                                             cv 51.7 %, var 0.2366
CL.median                 0.039525
CL.sdlog                   0.17642                                             cv 17.8 %, var 0.03112
ke                        0.080454
conc.additive              0.71712

standard errors not computed: the fit was stopped before it converged.

shrinkage (1 - SD of the units' scores): ka -23.4 %, CL 3.7 %

doctor: 2 notes
note     stopped: the fit was stopped before it ended, by the time limit (options timeout, 20 s): the findings are about where it stopped.
note     optima: one start: other optima were not looked for.
         what to do: options starts=8 looks from 8 starts.
```

The fit ran 28 seconds: the step under way when the limit passed was finished first.

## Options

In the fit text, one `options` line:

```text
options starts=4 seed=1 rtol=1e-8
```

or in Python, `problem.options(starts=4, seed=1, rtol=1e-8)`. Every option and its default:

| Option | Default | Means |
|---|---|---|
| `starts` | 1 | how many starts: start 1 is the values written, the others are drawn around them |
| `seed` | 0 | the random draws: the starts, the doctor's gradient check, NUTS's chains. The same seed, the same draws |
| `rtol` | 1e-6 | the solver's relative tolerance |
| `atol` | TenQSP's (1e-6) | the solver's absolute tolerance; also the level below which predictions in a transformed link (`log10(bact)`) are floored |
| `maxiter` | 200 | the most steps of each start's optimizer (and of each profile point's refit) |
| `progress` | `plot` in a notebook, `text` elsewhere | see above |
| `checks` | `full` | `quick` leaves out the doctor's tolerance and gradient checks |
| `timeout` | off | a time limit, as above |

`method nuts` has its own options on the method line: `method nuts chains=4 warmup=1000
samples=1000 target_accept=0.9` (those are the defaults; see [NUTS](../methods/nuts.md)).

`problem.summary()` lists the options, written or default:

```text
options    rtol 1e-6 (default), starts 8, seed 3, maxiter 200 (default), checks full (default)
```

An option that does not exist is an error, with the nearest name:

```text
FitTextError The fit text has 1 problem:

line 14: options start=4
options: unknown option 'start'. Did you mean 'starts'? Options: rtol, atol, starts, seed, maxiter, progress, checks, timeout.
```

### Starts

The other starts are drawn around the written values, on each quantity's estimation scale:
on the log scale, within about a factor 5 either way (95 %); within bounds, with an SD of 1 on
the logit scale; otherwise with an SD of half the value (at least 0.5). A start where the model
does not run is drawn again (`redrawn` in the printout counts them). Each start runs in its own
thread, on its own core: on a machine with enough cores, 8 starts take little longer than one.

More starts find optima that one start can miss; the doctor reports how many reached the best
fit (see [the doctor](doctor.md#several-optima)). For a new model, `starts=8` is a good
first fit.

### The tolerance

`rtol` 1e-6 is enough for most PK and PD models. Counts that span several orders of magnitude
(bacteria, viral loads, tumour cells), or a model with fast and slow processes, may need
`rtol=1e-8`. The doctor's tolerance check tells: it computes −2LL again at `rtol` / 100 and
reports a problem if anything that matters changes.

## The solver

A fit solves the model with TenQSP's BDF solver in JAX, compiled once for all units and all
steps, with exact derivatives of every output. The model is not recompiled when values
change, only when the problem changes.

**The step limit.** A trial point far from the data (an optimizer may try a rate of 10¹³³) can
need millions of solver steps. So each solve between two doses or events may take at most 50
times the most steps any unit takes at the starting values, but at least 5000 and at most
100 000. A run that needs more fails at once: the optimizer refuses that point and steps back,
and the failure is counted in `failed runs during the fit`. `tf.objective(problem).max_steps`
gives the limit.

A run that fails for any other reason (the solver cannot proceed) is treated the same way:
there is no fallback to another solver inside a fit. The one-off runs of the model the passport
makes (TenQSP's `diagnostics()` at the estimates) use TenQSP's SciPy backend, which retries a
segment where BDF fails with Radau.

## How long it takes

| Part | Theophylline (2 states, 12 subjects) | Infection age (102 states, 1 experiment) |
|---|---|---|
| until the first step (compiling) | about 15 s | about 15 s |
| the fit | 41 s (`laplace`, 1 start) | about 9 min (`optimize`, 4 starts) |
| the doctor's checks | a few seconds | about a minute |
| a profile | about 3 min (about 1 min with `scan=False`) | longer: each point is a fit |
| a VPC, 200 studies | about 10 s | |

Some advice:

- **Look before fitting.** `problem.summary()` costs nothing and catches a wrong link or a
  misread column.
- **Try a quick fit first**: `options checks=quick`; then the full fit, with several starts.
- **Starts run side by side**, so more starts cost little more time on a machine with several
  cores.
- **Profiles cost a fit per point.** Profile the quantities that matter; `scan=False` for a
  first look, the scan for the report.
- **A slow fit is often a fit on a ridge**: the optimizer walks along a valley the data do not
  close. The [QSP quickstart](../start/quickstart-qsp.md) shows one. More iterations will not
  help; the doctor's ridge finding says what will.
