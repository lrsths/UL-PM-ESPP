import torch
import numpy as np
import random


def decode_sampled_path_edge(
        G,
        edge_list,
        edge_probs,
        source: int, sink: int,
        *,
        num_samples: int = 50,
        min_prob: float = 1e-8,
        eps_greedy: float = 0.0,
        seed: int | None = None,
        verbose: bool = False):
    """
    Edge‑level probabilistic decoding for the shortest‑path task
    (best path is selected after multiple samples).

    Parameters
    min_prob : float
        Edges whose probability is below this threshold are ignored;
        setting it too low can leave the walker with no valid moves.

    eps_greedy : float
        If the *sum* of probabilities on all outgoing edges at the
        current node is < eps_greedy, fall back to the cheapest
        (lowest‑weight) edge as a last‑resort move.
    """

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    edge_probs = edge_probs.detach().cpu().numpy()
    p_dict = {tuple(e): float(p) for e, p in zip(edge_list, edge_probs)}

    if verbose:
        miss = [e for e in G.edges if e not in p_dict]
        print(f"[decode] missing edges in p_dict: {len(miss)}")

    best_path, best_cost = None, float("inf")

    for _ in range(num_samples):
        path, visited = [source], {source}
        cur = source
        dead = False

        while cur != sink:
            # out_edges = [(cur, v) for v in G.successors(cur)]
            # probs     = np.asarray([p_dict.get(e, 0.0) for e in out_edges])
            out_edges = [(cur, v) for v in G.successors(cur)]
            probs = np.asarray([p_dict.get(e, 0.0) for e in out_edges], dtype=float)

            visited_mask = np.asarray([v not in visited for _, v in out_edges], dtype=bool)
            valid_mask = np.logical_and(probs >= min_prob, visited_mask)

            # valid_mask = (probs >= min_prob) & np.array([v not in visited for _, v in out_edges])
            if not valid_mask.any():
                dead = True
                break

            out_edges = [e for e, m in zip(out_edges, valid_mask) if m]
            probs = probs[valid_mask]

            if probs.sum() < eps_greedy:
                nxt = min(out_edges, key=lambda e: G[e[0]][e[1]].get('weight', 0))[1]
            else:
                probs = probs / probs.sum()
                nxt = random.choices([v for _, v in out_edges],
                                     weights=probs, k=1)[0]

            path.append(nxt)
            visited.add(nxt)
            cur = nxt

        if not dead and cur == sink:
            cost = sum(G[u][v]['weight'] for u, v in zip(path[:-1], path[1:]))
            if cost < best_cost:
                best_cost, best_path = cost, path

    return best_path if best_path is not None else ["No valid path found"]
