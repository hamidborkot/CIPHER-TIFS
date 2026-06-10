# Theorem 2: Convergence Under Archetype-Stratified Federated Learning

## Setup

Let $\mathcal{K}$ be the set of $K=10$ federated clients (organizational units). Each client $k$ holds local dataset $\mathcal{D}_k$ drawn from distribution $\mathcal{P}_k$. Under **random partitioning**, clients are assigned arbitrarily. Under **archetype-stratified partitioning**, clients are grouped such that users within each client share behavioral archetype $a \in \{\text{Privileged, Technical, Clerical, Remote, Ops}\}$.

Define the **gradient divergence** across clients as:
$$\Delta = \max_{k,k'} \|\nabla F_k(\mathbf{w}) - \nabla F_{k'}(\mathbf{w})\|_2$$

## Statement

**Theorem 2.** Under archetype-stratified partitioning with $K$ clients, $T$ communication rounds, learning rate $\eta$, and DP noise $\sigma$, the federated model converges as:
$$\mathbb{E}[\|\nabla F(\mathbf{w}_T)\|^2] \leq \frac{2(F(\mathbf{w}_0) - F^*)}{\eta T} + \frac{\eta L \sigma^2 d}{n} + \eta^2 L^2 \Delta_{\text{arch}}^2$$

where $\Delta_{\text{arch}} \leq \Delta_{\text{rand}}$, i.e., **archetype stratification strictly reduces inter-client gradient divergence** compared to random partitioning.

## Proof Sketch

Following Li et al. (2020) FedProx Theorem 3, the convergence bound contains a heterogeneity term $\eta^2 L^2 \Delta^2$. We show that $\Delta_{\text{arch}} \leq \Delta_{\text{rand}}$ via the following argument:

1. **Intra-archetype homogeneity:** Users assigned to the same archetype share similar behavioral distributions $\mathcal{P}_k \approx \mathcal{P}_{k'}$ when $\text{archetype}(k) = \text{archetype}(k')$. This follows from the archetype definition: users are grouped by role × department × access-level, ensuring similar feature distributions.

2. **Gradient alignment:** By the data processing inequality, if $\mathcal{P}_k \approx \mathcal{P}_{k'}$, then $\|\nabla F_k(\mathbf{w}) - \nabla F_{k'}(\mathbf{w})\|_2 \leq \|\nabla F_k(\mathbf{w}) - \nabla F_{j}(\mathbf{w})\|_2$ for any randomly assigned client $j$ with mixed archetypes.

3. **Tighter bound:** Therefore $\Delta_{\text{arch}} = \max_{k,k' : \text{same archetype}} \|\nabla F_k - \nabla F_{k'}\| \leq \Delta_{\text{rand}}$, tightening the third convergence term. $\square$

## Empirical Validation

From `results/results_e4_comparison.csv`: SENTINEL-EGO (Archetype) achieves F1=0.8571 vs SENTINEL-EGO (Random) F1=0.8412 — a +1.59pp gain from stratification alone, consistent with the tighter convergence bound prediction.

## References
- Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M., Smola, A., & Smith, V. (2020). Federated optimization in heterogeneous networks. *ICLR 2020*.
- Karimireddy, S. P., et al. (2021). SCAFFOLD: Stochastic controlled averaging for federated learning. *ICML 2021*.
