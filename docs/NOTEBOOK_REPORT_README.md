# Consolidated Report Notebook

Read [CONSOLIDATED_REPORT_EXECUTED.ipynb](CONSOLIDATED_REPORT_EXECUTED.ipynb)
in VS Code or Jupyter. This English consolidated report contains 29 cells,
all 14 code cells executed successfully in a fresh kernel, and 35 embedded
figures. Its tables and images are readable without raw datasets or weights.

[CONSOLIDATED_REPORT.ipynb](CONSOLIDATED_REPORT.ipynb) is the editable source.
Re-execution requires the existing experiment exports under `runs/` and Python
with pandas, NumPy, matplotlib, PyYAML, scikit-learn, PyArrow and IPython/Jupyter.
No cell trains models, refits calibration or generates new experimental schedules.

The `report_artifacts/` folder contains 50 report tables in CSV/HTML, 24 derived
figures in PNG/PDF, 11 byte-identical original study figure copies, an export
index and source checksums. All 1,845 consumed source files remained unchanged
during export. Full per-seed, per-condition, per-class, calibration, bootstrap
and replay tables are included rather than replacing them with headline means.

Report sections distinguish clean experiments, partial robustness, baselines,
group-bootstrap uncertainty, calibration fallbacks and recorded replay evidence.
Predictive qualification remains not qualified, and incomplete scope is disclosed.