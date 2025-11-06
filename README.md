# causal-ai-fairness
This repo contains an operationalization of causal fairness concepts in AI. Synthetic and real-world experiments are included.

# 1. Basic survey
Everything has to be contextualized based on SFM:
1. Deline an exhaustive list of effects coming from causal literature, in particular the ones coming from causal fairness works. (partially)
2. In a similar manner, deline the normative criterions that are used to define "fairness" in causal terms (demographic parity, eq of odds, etc.)

# 2. Syntethic experiment
1. Given binary variables and the Markovian SFM, produce a dataset of observations.
2. Compute all effects that are coming from literature (identifiability equations)
   
# 3. Real-world experiment
For each dataset in the ones used in AEQUITAS (resources can be found https://aiod.eu):
1. Intepretazione dell'analisi osservazione in termini causali
2. Calcolare effetti causali

# 4. Experiments
1. Extension to multi-varied discrete variable (>2 X, >2 Y). The idea is to take inspiration from works that discussed extension to PN, PS and PNS.
2. Extension to multi-varied discrete (ordered) variable
3. Extension to multi-varied continuos variables
4. Extension to NON-linear models
