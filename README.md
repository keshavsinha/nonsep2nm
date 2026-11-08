# nonsep2nm — Functional and statistical non-separability of P, V and T in 2 nm class timing sign-off

Code, SPICE decks, netlists and results for the article

> S. Choudhury, K. Sinha, A. Kumar, "Functional and Statistical Non-Separability of Process, Voltage and
> Temperature in 2 nm Class Timing Sign-off: A SPICE-Calibrated Monte Carlo Study on Open Benchmarks"
> (submitted).

The framework synthesises open benchmarks to the ASAP7 predictive library, places them, and runs
full-graph Monte Carlo static timing (rise/fall NLDM arcs, clock-tree latency) with a device model fitted to
BSIM-CMG SPICE, placement-dependent IR drop and temperature, correlated fast/slow workload and
density-dependent EUV delay tails. It compares additive, RSS and simultaneous corners with the coupled
Monte Carlo quantile (functional separability) and coupled with mechanism-independent Monte Carlo
(statistical separability).

Every number in the paper can be regenerated from the JSON files in `results/`.

## Repository layout

| Path | Content |
|---|---|
| `nsim/` | Framework: Liberty reader, netlist and logic simulation, SimPL-style placement and RUDY, rise/fall NLDM STA, vectorised Monte Carlo STA with clock latency, device model, PDN and thermal grids, experiments |
| `syn/` | Timing-driven Yosys/ABC script (ORFS `abc_speed`) and the mapped ASAP7 netlists used in the paper |
| `rtl/` | Benchmark RTL (see `rtl/README.md` for sources and licences) |
| `spice/` | ngspice FO4 testbench, ASAP7 model-card converter, V/T/corner sweep, alpha-power fit and its results |
| `ot/` | OpenTimer v2 cross-check inputs for GCD |
| `prep_activity.py`, `prep_all2.py` | Activity simulation and design preparation (placement, timing graph, maps) |
| `r2_core.py`, `r2_sweeps.py`, `r2_dtco.py`, `r2_tornado.py`, `r2_finish.py` | Experiment drivers |
| `figs.py`, `figs3.py` | Figure scripts |
| `results/` | JSON summaries behind every table and figure |
| `figs/` | Figures as used in the paper |

### Where each result comes from

| Paper item | Script | Result file / key |
|---|---|---|
| Table 2 | `prep_all2.py` | `results/designs.json` |
| Fig. 2, device fit | `spice/sweep.py`, `spice/fit.py` | `results/spice_fo4.json`, `spice/fit.json` |
| Tables 4 and 5, Fig. 4 | `r2_core.py` | `results/core.json` (`<design>_<node>`; `stress_*` for 40k-sample stress runs) |
| Fig. 5, Fig. 6 | `r2_sweeps.py` | `results/sweeps.json` (`rho|…`, `V|…`, `IR|…`) |
| Fig. 8 | `r2_finish.py sweeps` | `results/sweeps.json` (`T|…`) |
| Fig. 7 | `r2_finish.py tornado` | `results/tornado.json` |
| Table 6 | `r2_dtco.py`, then `r2_finish.py dtco` (AES) | `results/dtco.json` |

## Requirements

* Python 3.10+ with the packages in `requirements.txt` (`pip install -r requirements.txt`)
* `yowasp-yosys` (installed by `requirements.txt`) to re-run synthesis
* ngspice 42 with OSDI support and OpenVAF-Reloaded to compile BSIM-CMG (only to repeat the SPICE sweep)
* OpenTimer v2 (only to repeat the cross-check)

## External data (not redistributed)

* ASAP7 NLDM libraries from OpenROAD-flow-scripts (`flow/platforms/asap7/lib/NLDM`): the RVT
  `SIMPLE`, `INVBUF` and `SEQ` libraries for TT (and SS, FF for reference). Copy or link them into `lib/` as
  `lib/SIMPLE_TT.lib`, `lib/INVBUF_TT.lib`, `lib/SEQ_TT.lib`.
* ASAP7 BSIM-CMG model cards (`asap7_pdk_r1p7/models/hspice`). `spice/mkmodel.py` converts them.
* BSIM-CMG Verilog-A (VA-Models repository), compiled with OpenVAF-Reloaded. ngspice 42 accepts OSDI 0.3
  only; the OSDI version field of the compiled library was set to 0.3 (the descriptor layout is backward
  compatible).

## Reproduce

Run from the repository root. Intermediate files go to `cache2/` (created automatically, not tracked).

```bash
# 0. (optional) re-synthesise with syn/syn2.sh (usage in its header; periods 310/380/950 ps for GCD/AES/RISC-V)
# 1. device model (optional; results are shipped)
cd spice && python3 mkmodel.py && python3 sweep.py && python3 fit.py && cd ..
# 2. activity and design preparation
python3 prep_activity.py
python3 prep_all2.py
# 3. experiments
python3 r2_core.py
python3 r2_sweeps.py
python3 r2_dtco.py
python3 r2_finish.py sweeps     # final tail runs (AES at 5,000 samples)
python3 r2_finish.py dtco       # AES DTCO at 6,000 samples
python3 r2_finish.py tornado    # sensitivity study (10,000 samples)
# 4. figures (read cache2/ if present, otherwise results/)
python3 figs.py fig_device2
python3 figs3.py fig_flow fig_maps fig_yield fig_rho fig_sweeps fig_tails fig_tornado
```

`fig_maps` and `fig_yield` need the prepared design and sample files in `cache2/` (steps 2 and 3); the other
figures can be drawn directly from `results/`.

All runs use fixed seeds and one random stream per mechanism, so separable and coupled runs share
realisations wherever their models agree.

OpenTimer cross-check: `python3 ot_export.py gcd` writes `ot/gcd.v` and `ot/gcd.timing`; run `ot/gcd.ot`
in OpenTimer, save the report, convert it with `python3 ot/parse_ot.py <report>`, and compare with
`python3 ot_compare.py gcd <report>.json`.

## Known limitations of the code

* Corners (`nsim.experiment.corner_stack`) are evaluated on the device function for a path of the nominal
  critical-path depth, not by corner STA of the netlist; they therefore depend on a design only through its depth.
* In the activity simulation the AES `rst` input (active low in the RTL) is held at 0, which keeps the 4-bit
  round counter `dcnt` in reset; the data path toggles normally. Activity only shapes the power maps, whose
  amplitude is normalised to the droop and thermal budgets.
* The alpha-power model uses a 0.02 V overdrive floor; results below about 0.55 V supply are less reliable.

## Licence

Code and data in this repository are released under the MIT licence (`LICENSE`). Third-party benchmark RTL
keeps its original licence (`rtl/README.md`).

## Citation

Please cite the article (see `CITATION.cff`). The archived release is available on Zenodo:
https://doi.org/10.5281/zenodo.XXXXXXX

## Use of AI assistance

Parts of this code were written with the help of an AI coding assistant (Claude, Anthropic). The authors
reviewed the code and verified the results against SPICE, OpenTimer and analytical references.
