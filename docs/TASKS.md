# Development task index

[Project board](https://github.com/users/Nada-belarbi/projects/2) · [Team workflow](../CONTRIBUTING.md)

41 implementation tasks transcribed from the five work packages. Individual assignments and dates are pending team selection. Dependencies below are proposed coordination prerequisites, not measured durations; mock interfaces may unblock parallel work. Validate them during kickoff.

## Member 1

[Technical guide](work-packages/member-1.md)

| Task | Issue | Prerequisites |
| --- | --- | --- |
| D01 — Contract fixtures | [#1](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/1) | None |
| D02 — Acquisition and audit | [#2](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/2) | None |
| D03 — Storage and validation | [#3](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/3) | [D01](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/1), [I02](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/35) |
| D04 — PAMAP2 adapter | [#4](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/4) | [D02](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/2), [D03](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/3) |
| D05 — MOSI adapter | [#5](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/5) | [D02](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/2), [D03](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/3) |
| D06 — Groups and decisions | [#6](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/6) | [D04](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/4), [D05](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/5), [E01](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/19) |
| D07 — Normalization | [#7](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/7) | [D06](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/6), [R01](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/27) |
| D08 — Dataset and batching | [#8](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/8) | [D07](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/7), [M01](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/10), [R01](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/27) |
| D09 — Handoff | [#9](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/9) | [D08](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/8) |

## Member 2

[Technical guide](work-packages/member-2.md)

| Task | Issue | Prerequisites |
| --- | --- | --- |
| M01 — Input contract | [#10](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/10) | None |
| M02 — Shared encoder | [#11](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/11) | [M01](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/10), [I02](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/35) |
| M03 — Source models and V0 | [#12](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/12) | [M02](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/11) |
| M04 — Trainer and persistence | [#13](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/13) | [M03](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/12), [D08](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/8), [I03](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/36) |
| M05 — Required baselines | [#14](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/14) | [M04](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/13) |
| M06 — Learned fusion V1 | [#15](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/15) | [M04](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/13) |
| M07 — Attention fusion V2 | [#16](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/16) | [M06](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/15) |
| M08 — Controlled experiments | [#17](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/17) | [M05](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/14), [M06](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/15), [M07](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/16), [E03](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/21), [E04](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/22) |
| M09 — Handoff | [#18](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/18) | [M08](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/17) |

## Member 3

[Technical guide](work-packages/member-3.md)

| Task | Issue | Prerequisites |
| --- | --- | --- |
| E01 — Protocol validator | [#19](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/19) | [D02](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/2) |
| E02 — Metrics and registry join | [#20](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/20) | [I02](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/35) |
| E03 — Saved scenarios | [#21](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/21) | [D01](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/1), [R01](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/27) |
| E04 — Evaluation runner | [#22](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/22) | [E01](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/19), [E02](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/20), [E03](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/21), [M04](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/13), [R02](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/28) |
| E05 — Calibration | [#23](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/23) | [E04](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/22) |
| E06 — Abstention | [#24](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/24) | [E05](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/23) |
| E07 — Comparisons | [#25](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/25) | [E06](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/24), [M08](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/17), [R07](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/33) |
| E08 — Reports and release | [#26](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/26) | [E07](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/25) |

## Member 4

[Technical guide](work-packages/member-4.md)

| Task | Issue | Prerequisites |
| --- | --- | --- |
| R01 — Shared selector | [#27](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/27) | [D01](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/1), [M01](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/10), [I02](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/35) |
| R02 — Stateless predictor | [#28](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/28) | [R01](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/27), [M03](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/12), [D08](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/8) |
| R03 — Source buffers | [#29](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/29) | [R01](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/27) |
| R04 — Stateful engine | [#30](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/30) | [R02](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/28), [R03](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/29) |
| R05 — Replay iterator | [#31](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/31) | [R04](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/30), [E03](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/21) |
| R06 — Application runs | [#32](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/32) | [R05](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/31), [D04](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/4), [D05](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/5), [M04](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/13) |
| R07 — Qualification | [#33](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/33) | [R06](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/32) |

## Member 5

[Technical guide](work-packages/member-5.md)

| Task | Issue | Prerequisites |
| --- | --- | --- |
| I01 — Project skeleton | [#34](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/34) | None |
| I02 — Contracts | [#35](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/35) | [D01](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/1), [M01](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/10) |
| I03 — Config resolver | [#36](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/36) | [I02](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/35) |
| I04 — Registries/services | [#37](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/37) | [I03](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/36) |
| I05 — CLI lifecycle | [#38](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/38) | [I04](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/37), [M04](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/13), [E04](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/22), [R05](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/31) |
| I06 — UI | [#39](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/39) | [I05](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/38), [R06](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/32) |
| I07 — CI/package | [#40](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/40) | [I06](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/39), [E08](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/26) |
| I08 — Handoff | [#41](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/41) | [I07](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/40), [D09](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/9), [M09](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/18), [R07](https://github.com/Nada-belarbi/multimodal-temporal-learning-under-missing-and-asynchronous-modalities/issues/33) |

## Start here

1. Assign the five responsibilities and review the shared contracts together.
2. Start I01 (package skeleton), D01 (data fixtures), M01 (model contract), and D02 (dataset audit) in parallel.
3. Freeze I02 contracts; build R01 selector, D03 storage, M02 encoder and E02 metrics against fixtures.
4. Complete the PAMAP2 V0 chain before expanding model variants. Audit MOSI access early.
5. Run robustness, calibration, replay parity and release qualification using the specified evidence gates.

A closed issue counts as completed only when its acceptance checklist and reviewed evidence are present. A task count is not a time or effort estimate.
