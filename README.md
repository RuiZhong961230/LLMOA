# LLMOA  

LLMOA: A novel large language model assisted hyper-heuristic optimization algorithm  


## Highlights  
• We propose a novel large language model assisted hyper-heuristic optimization algorithm.  
• The standard prompt engineering-guided Gemini is employed as the high-level component.  
• The low-level heuristics (LLHs) consist of ten well-designed search operators  
• Experiments on CEC2014, CEC2020, CEC2022, and 10 engineering problems are conducted.  

## Abstract
This work presents a novel approach, the large language model assisted hyper-heuristic optimization algorithm (LLMOA), tailored to address complex optimization challenges. Comprising two essential components – the high-level component and the low-level component – LLMOA leverages the LLM (i.e., Gemini) with prompt engineering in its high-level component to construct optimization sequences automatically and intelligently. Furthermore, we propose novel elite-based local search operators as low-level heuristics (LLHs), which draw inspiration from the proximate optimality principle (POP). These local search operators cooperated with well-known mutation and crossover operators from differential evolution (DE), at a total of ten efficient and versatile search operators, forming the whole LLHs. To assess the competitiveness of LLMOA, we conducted comprehensive numerical experiments across CEC2014, CEC2020, CEC2022, and ten engineering optimization problems, benchmarking against eleven state-of-the-art optimizers. Our experimental findings and statistical analyses underscore the powerfulness and effectiveness of LLMOA. Moreover, ablation experiments reveal the pivotal role of integrating the LLM Gemini and prompt engineering as the high-level component. Conclusively, this study provides a feasible avenue to introduce LLM to the evolutionary computation (EC) community. The research’s source code is available for download at https://github.com/RuiZhong961230/LLMOA.

## Citation
@article{Zhong:25,  
title = {LLMOA: A novel large language model assisted hyper-heuristic optimization algorithm},  
journal = {Advanced Engineering Informatics},  
volume = {64},  
pages = {103042},  
year = {2025},  
issn = {1474-0346},  
doi = {https://doi.org/10.1016/j.aei.2024.103042 },  
author = {Rui Zhong and Abdelazim G. Hussien and Jun Yu and Masaharu Munetomo},  
}

## Datasets and Libraries
CEC benchmarks are provided by the opfunu library, engineering problems are provided by the enoppy library, and gemini-pro API is provided by the google.generativeai library.
