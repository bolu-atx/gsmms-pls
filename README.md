# Growing structure Multiple Model Systems with Partial Least Squares Unit Models

## Abstract

Data-driven soft sensors have seen tremendous development and adoption in both academia and industry. However, one of the challenges remaining is modeling process drifts, degradation and discontinuities in steady-state. Since processes are never truly operating at a steady-state, it is often difficult to assess how much and what types of process data are needed for training and model maintenance in the future. A purely adaptive model maintenance strategy struggles against discontinuities such as preventive maintenance or catalyst changes. In mixture modeling and multi-model systems, the overall modeling structure is fixed and only local coefficients are adapted. In addition, multiple model systems require large amount of training data to initialize. In this paper, we propose an adaptive multiple model system utilizing growing self organizing map to model processes with drifts and discontinuities. Simple model update mechanisms such as recursive model update or moving window model update is not sufficient to deal with discontinuities such as abrupt process changes or grade transitions. For these scenarios, our approach combines projection based local models (Partial Least Squares) with growing self-organizing maps to allow for flexible adjustments to model complexity during training, and also later in online adaptation. This flexible framework can also be used to explore new datasets and rapidly develop model prototypes. We demonstrate the effectiveness of our proposed method through a simulated test cases and an industrial case study in predicting etch rate of a plasma etch reactor.

## Citing this work
The algorithm is published in Journal of Process Control.
```
@article{LU201856,
title = "Data-driven adaptive multiple model system utilizing growing self-organizing maps",
journal = "Journal of Process Control",
volume = "67",
pages = "56 - 68",
year = "2018",
note = "Big Data: Data Science for Process Control and Operations",
issn = "0959-1524",
doi = "https://doi.org/10.1016/j.jprocont.2017.06.006",
url = "http://www.sciencedirect.com/science/article/pii/S0959152417301191",
author = "Bo Lu and John Stuber and Thomas F. Edgar",
keywords = "Multiple model systems, Adaptive model, Growing self-organizing map, Growing structure multiple model systems, Soft sensor, Data-driven modeling, PLS, PCA, Projection to latent structure, Partial least squares",
}%
```
