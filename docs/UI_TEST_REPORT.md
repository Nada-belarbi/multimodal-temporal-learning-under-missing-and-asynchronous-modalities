# Reviewer Interface Test Report

Date: 31 October 2026. Tested application: Streamlit MMT Reviewer.

## Overall Result

The corrected reviewer is functional for saved-artifact audit, comparison,
step-based replay and scoped export. This is not complete release qualification,
nor a claim of production readiness. The live tested instance is
`http://127.0.0.1:8502`; the pre-existing 8501 instance returned HTTP 500 and was
left untouched. The new server health endpoint returns `ok`.

The current suite passes 124 tests under Python 3.11 with the UI extra; Ruff
passes. Streamlit AppTest exercises real artifacts and isolated navigation fixtures.
The bare-mode ScriptRunContext warning during AppTest is not an application exception.

## Defects Corrected

1. Artifact discovery only examined immediate directories. The reviewer now
   discovers nested completed exports and replays, ignores hidden/incomplete
   outputs, checks root containment, and skips invalid JSON metadata.
2. Preparation discovery missed the nested PAMAP2 path and exposed the earlier
   invalid MOSI profile. Both valid prepared application profiles are now available;
   incompatible class-map preparations are excluded.
3. Default config paths referred to absent templates or the wrong application.
   Replay exports resolve a matching real checkpoint and architecture-specific config.
   The `mmt ui` default prepared root now includes both applications.
4. Multi-checkpoint replay files were mixed when navigating a sequence. A separate
   checkpoint selector now isolates one run before selecting the sequence.
5. Previous/Next disabled states lagged after clicks. Callbacks update the index
   before rendering; first/last boundaries and Reset are verified.
6. Replay export attempted to serialize the entire multi-model dataset. It now
   exports one selected checkpoint/sequence, preserving selected observation IDs
   and excluding target labels, sentiment and group fields.
7. Headless export commands omitted original calibration/policy artifacts. These
   are now included when uniquely matched. A real MOSI segment rerun proved parity.
8. Deprecated `use_container_width` calls were replaced with supported width options.
   Comparison adds application/partition/scenario filters and a saved-metric chart.

## Verified Functional Paths

| Path | Evidence | Result |
| --- | --- | --- |
| Startup/render | New HTTP server and AppTest | No application exceptions |
| Data discovery | Existing artifacts and nested fixtures | 284 evaluation exports; 9 completed replay folders; 2 valid preparations |
| Audit | Select PAMAP2 and MOSI prepared roots | Source dimensions and sequence audit render |
| Comparison | PAMAP2, test and clean filters | Existing metrics filter/render; no metric refit |
| Replay | Whole-recording PAMAP2 shared-proof/clean | subject102 and subject106 available; nine checkpoint choices |
| Navigation | AppTest plus browser interaction | Next, Previous, Reset and boundary states work |
| Sequence/checkpoint change | Real-data AppTest | No exception; independent navigation index |
| No-data MOSI replay | Offline positive-delay trace | unavailable_no_data, null probabilities and hidden target fields |
| Export payload | Unit checks and real widgets | Target fields excluded; selected IDs retained; NaN displayed as null |
| Export command | Matching saved config/checkpoint/postprocessing | Valid paths; path escape rejected |
| UI/headless parity | New `runs/ui-validation-20261031-mosi/` export | IDs, scores, probabilities and status match original selected segment |
| Malformed JSON | Isolated fixtures | Corrupt metrics ignored; non-object manifest returns an explicit error |

Browser interactions covered all four tabs at measured CSS viewports
1440 x 1000 and 390 x 844. On actual PAMAP2 subject102, Next moved from decision
00000000 at 10.64 seconds to 00000001 at 11.64; Previous and Reset returned to
the first decision. No pageerror, visible Streamlit exception or control overlap
was detected during those interactions, and no document-wide horizontal overflow
was reported. Data grids intentionally support internal scrolling.

The UI-generated clean MOSI export command was actually executed in a separate
new output directory, without retraining or overwriting study artifacts. Its
selected IDs and status matched exactly; scores and probabilities matched at
atol=1e-6, rtol=1e-5. Non-clean reproduction remains disabled without the saved
schedule path. Ambiguous calibration lookup does not silently select a calibrator.

## Browser Evidence Limits

- Download buttons were rendered and exercised, and export payload construction
  was tested. The integrated browser did not confirm native download events or
  saved downloaded files; browser download completion is therefore not certified.
- Captures under `docs/ui_test_artifacts/` were inconsistent with the requested
  dimensions and sometimes stale. They are retained as diagnostics, **not reliable
  desktop/mobile screenshot proof**. Interaction/layout observations are separate.
- The reviewer is step-based and reads recorded artifacts. Continuous playback,
  pacing, browser-level parity across all recorded sequences, broad browser/OS
  compatibility and production deployment/security qualification were not tested.
- Large artifact folders remain an I/O/memory cost; scoped exports reduce the
  payload but this report does not claim a browser-memory performance SLA.

## Reproduction

```sh
uv run --locked --extra dev --extra ui python -m unittest discover -s tests -p test_ui.py -v
uv run --locked --extra dev --extra ui python -m unittest discover -s tests
uv run --locked --extra dev --extra ui ruff check src tests
uv run --locked --extra ui mmt ui --port 8502
```

The current tested server is already running on 8502; do not launch a second
instance on the same port. Use the URL above or another free port.