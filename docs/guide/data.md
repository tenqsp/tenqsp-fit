# The data

The data are one table: a CSV file (comma, tab or space separated; `.` is a missing value,
as in NONMEM) or, from Python, a pandas DataFrame. Every row is a measurement, a dose, or
neither. The fit text says which column is which, where the doses go, and which model output
each measurement measures. This page goes through those lines.

| Line | Means |
|---|---|
| `data "file.csv" with ID=Subject, TIME=Time` | the table, with its own column names mapped to the standard ones |
| `unit ID` | the column naming the experimental unit (subject, animal, flask, well) |
| `doses ...` | where the doses go |
| `observe conc = DV where CMT == 2` | which rows measure which output |
| `ignore DV where DVID == 3` | measurements left out on purpose |
| `censor conc below LLOQ` | measurements below a limit of quantification (M3) |

## Column names

The standard columns are NONMEM's: `ID`, `TIME`, `DV`, `AMT`, `EVID`, `MDV`, `CMT`, `DVID`,
`RATE`, `II`, `ADDL`, `SS`. A file that uses other names maps them in the `data` line, as
NONMEM's `$INPUT` does:

```text
data "theoph.csv" with ID=Subject, TIME=Time, DV=conc
```

Columns not mapped keep their names. Any column can be read by a `where` condition, and a
column used in a formula is a covariate (below). A `TIME` column is required; the times start
at 0 or later, and within a unit they go forward in the order of the file.

## Which rows are measurements and doses

- With an `EVID` column: `EVID 0` is a measurement, `EVID 1` a dose, `EVID 2` neither.
- Without it: a row with a `DV` value is a measurement unless `MDV` is 1 or it carries a dose
  amount; a row with a non-zero `AMT` is a dose.

The data can therefore be in R's layout (one row per measurement, the dose in a column), in
NONMEM's layout (dose rows and measurement rows), or in a wide layout (one column per kind of
measurement).

## The R layout: the dose as a column

R's `Theoph` data have one row per measurement and the dose (mg/kg) in a column `Dose`:

```text
"Subject","Wt","Dose","Time","conc"
1,79.6,4.02,0,0.74
1,79.6,4.02,0.25,2.84
...
```

```text
data "theoph.csv" with ID=Subject, TIME=Time, DV=conc
unit ID
doses Dose at t=0 into depot
observe conc = DV
```

`doses Dose at t=0 into depot`: each unit gets one bolus of its `Dose` value at time 0 into
the compartment `depot`. The dose column must have one value per unit.

## The NONMEM layout: dose rows

The same data with dose rows, `CMT 1` for the dose and `CMT 2` for the samples, and the
weight as `WT`:

```text
ID,TIME,AMT,DV,EVID,MDV,CMT,WT
1,0.0,4.02,.,1,1,1,79.6
1,0.0,.,0.74,0,0,2,79.6
1,0.25,.,2.84,0,0,2,79.6
1,0.57,.,6.57,0,0,2,79.6
...
```

```text
data "theoph_nonmem.csv"
unit ID
doses CMT 1 into depot
observe conc = DV where CMT == 2
```

`doses CMT 1 into depot` maps CMT numbers to the model's compartments by name; several are
written `doses CMT 1 into depot, 2 into centr`. When every dose goes to one place,
`doses into depot` is enough and the CMT column is not read for doses.

`summary()` shows how the rows were read:

```text
theoph_nm: model theoph (2 states), data "theoph_nonmem.csv": 12 units, 132 observations, 12 doses
units      ID: 12 units
links      conc <- DV where CMT == 2   132 rows, 12 units
doses      AMT by CMT: 1 -> depot   12 rows
...
```

Dose records are read by TenQSP: boluses, infusions (`RATE`, or `DUR` for the duration),
and repeated doses (`ADDL` with `II`). A record of an infusion of 100 at rate 50, repeated
twice every 12 h, is read as three infusions (with `doses into centr`):

```text
ID,TIME,AMT,RATE,ADDL,II,EVID,DV
1,0,100,50,2,12,1,.
1,1,.,.,.,.,0,3.1
1,13,.,.,.,.,0,4.0
1,25,.,.,.,.,0,4.4
```

```python
print(p.units()[0])
print(p.table())
```

```text
<unit 1: 3 doses, 3 times; conc 3>
   ID  TIME  EVID    AMT  RATE    CMT  DVID   DV
0   1   0.0     1  100.0  50.0  centr   NaN  NaN
1   1   1.0     0    NaN   NaN    NaN  conc  3.1
2   1  12.0     1  100.0  50.0  centr   NaN  NaN
3   1  13.0     0    NaN   NaN    NaN  conc  4.0
4   1  24.0     1  100.0  50.0  centr   NaN  NaN
5   1  25.0     0    NaN   NaN    NaN  conc  4.4
```

Not read yet, each with a message saying what to do instead: steady-state doses (`SS`), resets
(`EVID 3` and `4`), a rate or duration set by the model (`RATE -1`, `-2`), doses into a field
of a structured population (doses go into compartments), and a measurement at the time of a
dose into a compartment it reads directly when the measurement row comes before the dose row
(NONMEM reads such a row as the value just before the dose; TenQSP gives the value just
after it).

Without a `doses` line, the model's own doses (if it has any) are used for every unit.

## Linking measurements to the model

Each kind of measurement is linked to an observable of the model by an `observe` line.
`where` picks the rows, by any columns:

```text
observe log10(bact) = DV where DVID == "bact"
observe log10(phage) = DV where DVID == "phage"
```

- The left side is an observable, or a formula of observables when the data are on another
  scale: `log10(bact)`, `log(conc)`. The error model then applies on that scale.
- A condition compares columns with numbers or quoted text (`==`, `!=`, `<`, `<=`, `>`,
  `>=`) and joins comparisons with `and`, `or`, `not`.
- Every measurement row must be linked once, or left out on purpose:
  `ignore DV where DVID == "phage"`. Two `observe` lines may not read the same row.

When measurements are not linked, the error proposes the lines to write. For the time-kill
data of the [QSP quickstart](../start/quickstart-qsp.md) without `observe` lines:

```text
The data are not linked to the model.
  DV rows by DVID:  "bact" -> 12 rows;  "phage" -> 12 rows
  observables of the model:  bact, phage
Write one line per kind of measurement, for example:
  observe bact = DV where DVID == "bact"
  observe phage = DV where DVID == "phage"
(or leave some out on purpose: ignore DV where DVID == "phage")
```

and for the NONMEM file above without `observe` and `doses` lines, besides that proposal:

```text
The data have 12 dose rows (AMT), but no doses line says where they go.
Write, for example:
  doses CMT 1 into depot
or, for every dose into one place:
  doses into depot
Compartments: depot, centr.
```

All problems of a fit text are reported together, each with its line.

### The wide layout

A table with one column per kind of measurement links each column by name:

```text
TIME,bact,phage,BLQ
0.5,6.217,4.038,0
1.0,6.478,4.26,0
...
```

```text
observe log10(bact) = bact
observe log10(phage) = phage
```

```text
links      log10(bact) <- bact (all rows)   12 rows   ...
           log10(phage) <- phage (all rows)   12 rows
```

A row is a measurement of a column when that column has a value and the row is not a dose.

## Below the limit of quantification

`censor` marks measurements below a limit. They enter the likelihood as the probability that
the value is below the limit (Beal's M3 method), not as values. The limit is on the link's
scale: for `log10(bact)`, `below 2` is 100 per mL.

| Line | Censored rows |
|---|---|
| `censor conc below 0.1` | those with DV strictly below 0.1 |
| `censor conc below LLOQ` | those with DV strictly below their own `LLOQ` column |
| `censor conc where BLQ == 1 below LLOQ` | those flagged by `BLQ`; their DV is not read |
| `censor conc where DV == "BLQ" below 0.1` | those whose DV is the text `BLQ` |

Censored values are often written as the limit itself. Then "strictly below" finds none, and
the check says so. For the time-kill data, where the three counts below 100 per mL are written
as 2:

```text
line 6: censor log10(bact) below 2
censor log10(bact): 3 measurements equal the limit exactly (rows 19, 21 and 23). If they are values below the limit written as the limit, say how to tell them: censor log10(bact) where CENS == 1 below 2 (a flag column). If they are measured values, write the rule out: censor log10(bact) where DV < 2 below 2.
```

Rows are counted from 1 after the header, as a spreadsheet shows them less one. With the flag
(`censor log10(bact) where BLQ == 1 below 2`), the summary counts them:

```text
links      log10(bact) <- DV where DVID == "bact"   12 rows   censored (M3, limit 2): 3 rows, where BLQ == 1
```

## Units

`unit ID` names the column of the experimental unit. Rows with the same value belong to one
unit; parameters written with a distribution vary between units (see
[estimates](estimates.md)). Without a `unit` line all rows are one unit, as in a single
experiment. A unit without linked measurements is not fitted, and the summary says how many
there are.

## Covariates

A data column used in a formula of the fit is a covariate, read once per unit:

```text
CL = CL_pop*(WT/70)^0.75
```

or inside a distribution, `estimate CL ~ lognormal(median=CL_pop*(WT/70)^0.75, cv=0.3)`.
A covariate must have one value per unit; values that change with time are not read yet.
Each unit's covariates are shown by `p.units()`:

```python
u = p.units()[0]
print(u)
print(u.covariates)
```

```text
<unit 1: 1 dose, 11 times; conc 11>
{'WT': 79.6}
```

## The data as the fit reads them

- `p.summary()`: every link with its row and unit counts, the doses, the units.
- `p.units()`: one `Unit` per unit, in the order they first appear, with its TenQSP scenario
  (doses and measurement times), its measurements by link (`Observations`: `times`,
  `values`, the data `rows`, and for a censored link `below` and `censored`) and its
  covariates.
- `p.table()`: the data as read, as one NONMEM-style table: a row per dose (with the
  compartment by name) and per linked measurement (`DVID` the link), covariates and, for a
  censored link, `LLOQ`. `p.table().to_csv("data.csv", index=False, na_rep=".")` writes it.
- `p.provenance()`: the file's path and SHA-256, the column map, how many rows were measured,
  linked, dosed and left out, and the versions. It is recorded with every fit, in the
  [passport](doctor.md).
