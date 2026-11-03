---
title: Hierarchical auditory predictions in the local–global paradigm
subtitle: Project overview
---

+++ {"part": "abstract"}
Local deviance (a change in the fifth sound of a quintlet) produced an early frontocentral MMN that localised to left auditory cortex and tracked a learner of sound-to-sound transitions. Global deviance (a rare quintlet in a block) produced a late parietal P3b that tracked a learner of the block rule. The two responses dissociated in both directions. In dynamic causal models, local surprise strengthened the forward A1 → STG connection and top-down input to STG. Global deviance propagated forward from STG to parietal cortex.
+++

## The question

The local–global paradigm {cite:p}`Bekinschtein2009` separates two levels of auditory prediction. **Local** predictions concern the next sound; **global** predictions concern the structure of the whole sequence. This project asks three questions of the open dataset from @Rutiku2024 (OpenNeuro [ds005429](https://openneuro.org/datasets/ds005429), n = 15):

1. Which EEG responses index each level, and do they interact? ([ERPs](site/results-erps.md))
2. What computation does each response follow: fixed categories, adaptation, or learning at the local or global level? ([Learner models](site/results-models.md))
3. Where are the responses generated, and how do they move through the auditory–frontal–parietal network? ([Sources](site/results-source.md), [DCM](site/results-dcm.md))

The work contributes to the Kikuchi lab's MRC programme on laminar feedforward and feedback signalling in predictive coding. It provides a human EEG reference for the same paradigm used in macaque and marmoset recordings.

## Key results

:::{list-table}
:header-rows: 1

* - 
  - Local deviance (LD − LS)
  - Global deviance (GD − GS)
* - ERP
  - Frontocentral MMN, 80–140 ms, −1.22 µV, *t*(14) = −9.4, 15/15
  - Parietal P3b, 300–700 ms, +2.35 µV, *t*(14) = 5.4, 15/15
* - Best model
  - Local transition learner (M2), PXP = 0.94
  - Global rule learner (M3), PXP = 0.95
* - Source (eLORETA)
  - Left Heschl's gyrus + STG, cluster *p* = .0005
  - No reliable focal source
* - DCM (0–250 ms)
  - Forward A1 → STG ↑, IFG/IPL → STG ↑
  - Forward STG → IPL ↑ (left)
:::

```{figure} figures/Figure1.png
:label: fig-overview
:width: 100%
Local and global deviance produce an early frontocentral MMN and a late parietal P3b. Details in [](site/results-erps.md).
```

## How to use this site

- **Results**: one page per figure, with every statistic, its controls, and what the data do *not* support.
- **[Methods](site/methods.md)**: dataset, preprocessing, the learner models (with equations), inference and the source/DCM set-up.
- **[Validation](site/validation.md)**: simulations that check each method returns a known answer.
- **[Analysis notebook](notebooks/results_analyses_executed.ipynb)**: re-derives every number in the Results from the saved derivatives, with outputs. It ends with a table matching all 62 reported values (62/62 reproduced).
- **[Pipeline](site/pipeline.md)**: which script produces what, and how to rerun it.

:::{note} Status
Working draft, not peer reviewed. Numbers will change as analyses are updated. Check the date at the top of each page.
:::
