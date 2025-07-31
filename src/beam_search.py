def beam_search(
        G: None,
        source: int,
        sink: int,
        *,
        beam_width: int = 1000,
        max_depth: int | None = None,
):
    """
    Heuristic beam search (elementary paths).
    Returns *one* best path & cost. If no path found → (None, inf).
    """
    max_depth = max_depth or G.number_of_nodes()
    max_sample = 5000
    beam_width = 1000
    count = 0
    # (cost_so_far, node, visited_mask, path)
    beam = [(0.0, source, 1 << source, [source])]
    best_path, best_cost = None, float("inf")

    for _ in range(max_depth):
        if not beam:
            break
        if count > max_sample:
            break

            # expand every partial path in the current beam
        next_beam = []
        for cost, u, mask, path in beam:
            for v in G.successors(u):
                if (mask >> v) & 1:  # already visited → skip
                    continue
                w = G[u][v]["weight"]
                new_cost = cost + w
                new_mask = mask | (1 << v)
                new_path = path + [v]

                if v == sink:
                    count += 1
                    if new_cost < best_cost:
                        best_cost, best_path = new_cost, new_path
                else:
                    next_beam.append((new_cost, v, new_mask, new_path))

        # keep only the cheapest `beam_width` partial paths
        next_beam.sort(key=lambda t: t[0])
        beam = next_beam[:beam_width]

    return best_path, best_cost
