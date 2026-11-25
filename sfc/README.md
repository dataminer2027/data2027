# sfc

Code for *Certified Temporal Logic Rules for Interpretable Time Series
Classification*. A rule network proposes a temporal-logic formula over a signal,
and the formula is then checked rather than asserted: a smooth lower bound is
used during training, and a separate exact monitor decides whether the rule
holds on a given series.

## Two things a rule has to get right

**What the symbols refer to.** A rule means something only relative to a view, a
unit and a set of sample indices. The strings `(r[t+1] - r[t]) > 0.5` and
`r[t+1] > 0.5` differ in one symbol and check different events, so a monitor
that only sees the string can certify an event the classifier never fired on.
Views here are typed, and a formula that mixes sorts is rejected before
evaluation.

**How a smooth score relates to a definite answer.** Training needs
differentiability, and a smooth score near zero can have the opposite sign from
the exact one. So `rtilde > 0` is treated as a sufficient condition for
`rho > 0`. It is never treated as equivalent, and the code reports both numbers.

## Install

```bash
pip install -e .
```

Depends on `numpy` only; tests use the standard library.

## Usage

```python
import sfc
from sfc.parity import context, parity_dataset, parity_rule

rule = parity_rule()
X, y, _ = parity_dataset(0.5)
k = next(i for i, label in enumerate(y) if int(label) == 1)   # a record labelled 1

cert = sfc.certify(rule, context(X[k]), beta=8.0)
assert cert.certified and cert.sound      # lower > 0 implies exact > 0
print(cert.exact, cert.lower, cert.gap)   # 0.5  0.0163  1.4517
```

The gap between the exact value and the bound is not small, so a satisfied rule
is often not certified. That is the expected behaviour, and it is visible in the
output:

```python
import sfc
from sfc.monitor import rho, rtilde
from sfc.parity import context, parity_dataset, parity_rule

rule = parity_rule()

for amplitude in (1.0, 0.3):
    X, y, _ = parity_dataset(amplitude)
    k = next(i for i, label in enumerate(y) if int(label) == 1)
    x = X[k]

    exact = rho(rule, context(x), 0)          # true robustness
    lower = rtilde(rule, context(x), 0, 8.0)  # differentiable lower bound

    assert lower <= exact                     # holds on every formula
    print(amplitude, exact, round(lower, 3), lower > 0)
```

```
1.0  1.0    0.516  True     # certified: the rule provably holds
0.3  0.3   -0.179  False    # holds, but the bound is too weak to certify
```

## Numbers this code produces

Sect. 5.6 and 5.7 state every parameter of the parity benchmark, so those runs
are reproduced here:

| Parity benchmark | Paper | This repository |
|---|---|---|
| `sigma = 0`: exact positives / certified firings | 384 / 192 | identical |
| `sigma = 0.20`: exact positives / certified / agreement | 218 / 130 / 88.5% | 220 / 133 / 88.7% |
| Until scan gap `<= 2 log T / beta` | Theorem 2 | holds in all 768 checks |

The `sigma = 0.20` row differs slightly because the paper does not specify its
noise draws. The sweep uses 16 shared draws per amplitude with
`numpy.random.default_rng(11)`.

Two counts in the paper deserve a comment. The "all 384 negative records stay
negative" check appears here as the `label flips` column of
`examples/parity.py`, which is zero at every `sigma`; that follows from the
parity rule's robustness being exactly `+a / -a`, since a flip needs a
perturbation larger than the amplitude, and the sweep stays well below it. The
48/48 decision count (8 records times 6 amplitudes) is deterministic and is a
unit test, not a measurement: at `sigma = 0` the rule is exact, so the only
failure it can catch is a bug in the monitor. The evidence about noisy data is
the sweep in the table above.

The ten-dataset UEA comparison is not reproduced. The datasets and the per-run
records are not part of this repository.

## Source map

```
sfc/
  views.py        typed views and sorts (Definition 1)
  codebook.py     shared cumulative-softmax centres (Definition 2)
  relax.py        smin/smax and the per-reduction gap (Lemma 1)
  monitor.py      exact and relaxed semantics for G/F/U/Run/Trend
  until_scan.py   linear-time sound Until (Theorem 2)
  certify.py      certificate tuple and split-conformal lower bound
  parity.py       cross-channel parity benchmark (Sect. 5.6/5.7)
docs/EQUATIONS.md  paper equation to function
```

## Tests

```bash
python -m unittest discover -s tests -v
```

Sixteen tests cover the analytic claims rather than the table: Lemma 1 and
Lemma 2 on one-sided bounds and the compositional gap, Theorem 2 comparing the
linear-time scan against direct evaluation on random series up to `T = 256`, the
soundness direction of Theorem 1 over random formulas, the exact semantics of
`G`, `F`, `U`, `Run` and `Trend`, sort checking from Definition 1, and the
`+a / -a` robustness of the parity rule.

## Limits

- The training objective of Eq. (16)-(17) is not implemented. What is
  implemented is the object it optimises, namely `rtilde` with its per-reduction
  and compositional bounds, plus the exact monitor that decides whether a firing
  was real.
- `until_scan` is sound but not tight. It returns a lower bound within
  `2 log T / beta` of direct evaluation, and that bound is measured in
  `examples/parity.py` rather than assumed.
- Conformal bounds in `certify` use a split-conformal quantile, so the
  calibration split is the caller's responsibility.

## Citation

The manuscript is under review. Until it appears, cite the repository:

```
@misc{sfc2026,
  title  = {sfc: semantic-fidelity certification for temporal logic rules},
  year   = {2026},
  note   = {https://github.com/<owner>/sfc}
}
```

## License

MIT. See `LICENSE`.
