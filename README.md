# Layered Shapley

Each directory contains a separate sampling procedure.

### Standard:
The full algorithm:

    - In each layer:
        - Compute m_k
        - Draw m_k coalitions
        - compute the MC in each coalition
        - average the MC
    - sum up the layers

### Monte Carlo
The Monte Carlo sampling algorithm as presented by Ghorbani et al. in Data Shapley, adapted:

    - Draw a permutation:
        - Use all datapoints before i as the coalition
        - compute Shapley Value for that coalition
        - loop until convergence

### Passwise Sampling
A combination of Monte Carlo sampling with the standard algorithm.  

    - Compute the m_ks
    - For each evaluated datapoint i:
        - In each iteration:
            - Draw a k with probability p
            - Draw a sample S from k with uniform probability
            - Compute the marginal contribution of i on S
            - Loop until convergence

### Convergence in each layer
    - Compute m_ks
    - In each layer, sample until convergence is reached or m_k coalitions have been sampled
    - move on
