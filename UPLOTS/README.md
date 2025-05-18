# UPLOTS: A Unified Pretrained Time-Series Language Model for Constrained TS Generation



> **Abstract:**In time-series generation, existing approaches typically handcraft or train a separate model for each dataset, which hinders their scalability and fails to leverage shared temporal structures across domains. To address this fragmentation, we propose UPLOTS, a \textbf{U}nified, \textbf{P}rompt-guided \textbf{L}anguage model framework f\textbf{O}r constrained \textbf{T}ime-series \textbf{G}eneration across diverse domains. Instead of building task-specific models, UPLOTS leverages a single pre-trained transformer backbone guided by learned constraint prompts, enabling on-demand generation with precise pattern control. One key innovation is our dynamic multi-dataset loss re-weighting and prompt-to-pattern mapping, which allows UPLOTS to internalize inverse temporal structures during training and conditionally generate them at inference. We evaluate UPLOTS on four real-world benchmarks (ETTh, Energy, PEMS04, and PEMS08) and two critical peak patterns, demonstrating superior fidelity, pattern compliance, and generalization against multiple SOTA baselines and single-dataset variants. The experiment results position UPLOTS as a flexible, scalable model for time-series synthesis, demonstrating the potential of a single prompt-driven LLM to learn, adapt, and generate across domains. 



## Acknowledgement

We appreciate the following github repos a lot for their valuable code base:

https://github.com/lucidrains/denoising-diffusion-pytorch

https://github.com/cientgu/VQ-Diffusion

https://github.com/XiangLi1999/Diffusion-LM

https://github.com/philipperemy/n-beats

https://github.com/salesforce/ETSformer

https://github.com/ermongroup/CSDI

https://github.com/jsyoon0823/TimeGAN
