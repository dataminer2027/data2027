# Paper to source

Where each definition, equation and theorem of the SFC manuscript is decided in
code.

| Paper | Meaning | Code |
|---|---|---|
| Def. 1 | sort discipline (Proposition 1) | `sfc.views.well_typed` |
| Def. 2 | shared cumulative-softmax centres | `sfc.codebook.SharedCodebook` |
| Eq. (1)-(2) | standard robustness, including `Until` | `sfc.monitor.rho` |
| Eq. (3) | semantic fidelity | `sfc.monitor.semantic_fidelity` |
| Eq. (4)-(5) | `smin`, `smax` | `sfc.relax.smin`, `smax` |
| Lemma 1 | one-sided bounds and `log n / beta` gap | `tests.test_sfc.OneSidedBounds` |
| Eq. (6)-(9) | relaxed `And`/`Or`/`G`/`F` | `sfc.monitor.rtilde` |
| Eq. (10) | compositional gap `k log T / beta` | `sfc.monitor.gap_bound`, `levels` |
| Eq. (11)-(13) | linear-time sound `Until` | `sfc.until_scan.until_scan` |
| Theorem 2 | scan is a lower bound within `2 log T / beta` | `tests.test_sfc.UntilScan` |
| Eq. (14) | `Run` (n consecutive satisfied positions) | `sfc.monitor.Run` |
| Eq. (15) | `Trend` | `sfc.monitor.Trend` |
| Eq. (16)-(17) | training objective | not implemented, see below |
| Theorem 1 | `rtilde > 0 => rho > 0` | `tests.test_sfc.Soundness` |
| Sect. 4.6 | certificate tuple and conformal bound | `sfc.certify.certify`, `conformal_lower_bound` |

## Absent on purpose

Eq. (16)-(17), the training objective. What the repository does provide is the
object that objective is defined against, `rtilde` with its per-reduction and
compositional bounds, and the exact monitor that decides whether a firing was
real.

The ten-dataset UEA comparison is absent as well: those datasets and the per-run
records are not part of the repository.

The 48/48 decision count of Sect. 5.6 exists only as
`tests.test_sfc.test_all_eight_records_are_correct`, which runs 8 records at 6
amplitudes with `sigma = 0`. It is a check on the monitor, not a measurement on
noisy data.
