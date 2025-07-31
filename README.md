### Unsupervised Learning for the Elementary Shortest Path Problem

#### Abstract
The *Elementary Shortest‑Path Problem* (ESPP) seeks a minimum cost path
from $s$ to $t$ that visits each vertex at most once. The presence of negative-cost cycles renders the problem $\mathcal{NP}$‑hard. We present a probabilistic method for finding near-optimal ESPP, enabled by an unsupervised graph neural network that jointly learns edge-selection probabilities and node value estimates via a surrogate loss function.
The loss provides a high probability certificate of finding near-optimal ESPP solutions by simultaneously reducing negative cycles and embedding the desired algorithmic alignment. At inference time, a decoding algorithm transforms the learned edge probabilities into an elementary path. Experiments on graphs of up to 100 nodes show that the proposed method surpasses both unsupervised baselines and classical heuristics, while exhibiting high performance in cross-size and cross-topology generalization on unseen synthetic graphs.


How to run this code:

```shell
 python ./main.py --graph-file graph/er_30_test.pt --model-file mdl/ESPP_NNAA_30.pt