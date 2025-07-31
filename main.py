"""
how to run this code:
    python ./main.py \
        --graph-file graph/er_30_test.pt \
        --model-file mdl/ESPP_NNAA_30.pt
"""

import time, random
import numpy as np
import torch
from torch_geometric.utils import to_networkx
from torch_geometric.data import DataLoader
from src.beam_search import beam_search
from src.decoder import decode_sampled_path_edge

torch.cuda.empty_cache()
import os

# from ablation.decoding_method import (
#     decode_greedy_path, decode_sampled_path
# )
#
# from ablation.decoding import (
#     decode_sampled_path,
#     decode_sampled_path_edge,
#     decode_sampled_path_edge_greedy,
#     decode_sampled_path_edge_beam,
#     decode_beam_search
# )
#
# from ablation.utils import (
#     labeling_setting_shortest_path, beam_search_best,
#     labeling_espp_strict, evaluate_paths
# )

from src.gnn import build_model

import argparse


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--graph-file", required=True,
                   help="data with .lp")
    p.add_argument("--model-file", default=".",
                   help="the file store best_*.pt")
    return p.parse_args()


def compare_decoded_paths(decoded_info, test_graphs, t0, lp_costs, lp_total_time, model_label="Model"):
    print(f"=== Compare Decoded Paths with Beam Search for {model_label} ===")

    correct_count = 0  # decode == ground-truth
    better_count = 0  # decode  < ground-truth        ### NEW
    total_count = 0
    better_count_rand = 0

    v_true, v_pred, ratio_list = [], [], []
    v_rand, ratio_list_rand = [], []  # new
    diff_sum, base_sum = 0.0, 0.0  # new
    diff_sum_rand, base_sum_rand = 0.0, 0.0
    diff_sum_lp, base_sum_lp = 0.0, 0.0
    epsilon = 1e-7

    bs_path_len_sum = 0.0  # sum of beam search path length
    bs_node_cnt_sum = 0  # sum of beam search number of nodes in the path
    dec_path_len_sum = 0.0  # sum of decoding path length
    dec_node_cnt_sum = 0
    rand_path_len_sum = 0.0
    rand_node_cnt_sum = 0

    for item in decoded_info:
        idx = item["graph_index"]
        path_dec = item["decoded_path"]
        path_rand = item["rand_path"]
        src, snk = item["source"], item["sink"]
        G_g = test_graphs[idx]

        # ground-truth via beam_search_best
        sp_nodes, sp_length = beam_search(G_g, src, snk, beam_width=200)
        v_true.append(sp_length)

        print(f"\n[Graph {idx}]")
        print(f"  Beam Search Path = {sp_nodes}, length = {sp_length:.2f}")

        # decoded path cost
        path_edges = list(zip(path_dec, path_dec[1:]))
        dec_len = sum(G_g[u][v]["weight"] for (u, v) in path_edges)
        path_edges_rand = list(zip(path_rand, path_rand[1:]))
        rand_len = sum(G_g[u][v]["weight"] for (u, v) in path_edges_rand)
        v_pred.append(dec_len)
        v_rand.append(rand_len)

        # counting gt and decode length and number of gt & decode
        bs_path_len_sum += sp_length
        bs_node_cnt_sum += len(sp_nodes)
        dec_path_len_sum += dec_len
        dec_node_cnt_sum += len(path_dec)
        rand_path_len_sum += rand_len
        rand_node_cnt_sum += len(path_rand)
        #

        print(f"  Decoded path      = {path_dec}, length = {dec_len:.2f}")

        # stats
        if abs(dec_len - sp_length) < epsilon:
            correct_count += 1
        if dec_len < sp_length - epsilon:  # decode strictly better ### NEW
            better_count += 1
            # >>> rand
        if rand_len < sp_length - epsilon:  # Random 0.5 better than beam
            better_count_rand += 1

        total_count += 1

        ratio_list.append(dec_len / sp_length * 100)
        ratio_list_rand.append(rand_len / sp_length * 100)

        diff_sum += dec_len - sp_length
        base_sum += abs(sp_length)

        # >>> rand
        diff_sum_rand += rand_len - sp_length  # >>> rand
        base_sum_rand += abs(sp_length)

        # >>> LP
        lp_len = 0
        diff_sum_lp += lp_len - sp_length
        base_sum_lp += abs(sp_length)

    # summary numbers
    better_rate = better_count / total_count if total_count else 0.0  # NEW
    better_rate_rand = better_count_rand / total_count if total_count else 0.0
    rmse = np.sqrt(np.mean((np.array(v_pred) - np.array(v_true)) ** 2))
    rmse_rand = np.sqrt(np.mean((np.array(v_rand) - np.array(v_true)) ** 2))

    avg_ratio_rand = np.mean(ratio_list_rand)
    overall_gap = diff_sum / base_sum * 100
    overall_gap_rand = diff_sum_rand / base_sum_rand * 100
    overall_gap_beam_lp = diff_sum_lp / base_sum_lp * 100

    # path length
    avg_bs_len = bs_path_len_sum / total_count if total_count else 0.0
    avg_bs_nodes = bs_node_cnt_sum / total_count if total_count else 0.0
    avg_dec_len = dec_path_len_sum / total_count if total_count else 0.0
    avg_dec_nodes = dec_node_cnt_sum / total_count if total_count else 0.0
    avg_rand_len = rand_path_len_sum / total_count if total_count else 0.0
    avg_rand_nodes = rand_node_cnt_sum / total_count if total_count else 0.0

    # extra stats (kept from original)
    neg_count = sum(1 for w in v_pred if w < 0)
    ground_neg_count = sum(1 for w in v_true if w < 0)
    neg_rate = neg_count / ground_neg_count if ground_neg_count > 0 else 0.0
    neg_count_rand = sum(1 for w in v_rand if w < 0)
    neg_rate_rand = neg_count_rand / ground_neg_count if ground_neg_count > 0 else 0.0

    # per-batch timing (needs global decoded_info_model1 list)
    total_t_decode = sum(d["time_decode"] for d in decoded_info)
    total_t_beam = sum(d["time_beam"] for d in decoded_info)
    total_t_rand = sum(d["time_rand"] for d in decoded_info)

    #  print summary
    print(f"\nRate decode < beam for {model_label}: {better_rate * 100:.2f}% "
          f"({better_count}/{total_count} graphs)")
    print(f"Random 0.5 better-than-beam rate  : {better_rate_rand * 100:.2f}%"
          f"({better_count_rand}/{total_count} graphs)")
    print(f"[Random 0.5]  RMSE {rmse_rand:.4f} | avg ratio {avg_ratio_rand:.2f}% "
          f"| overall gap {overall_gap_rand:.2f}%"f"|negative rate{neg_rate_rand * 100:.2f}%")

    print(f"RMSE                                   : {rmse:.4f}")
    print(f"Overall Optimality Gap                 : {overall_gap:.4f}%")
    print(f"Number of decoded paths with neg weight: {neg_count} "
          f"({neg_count}/{ground_neg_count})")
    print(f"Decoded negative-weight rate           : {neg_rate:.2%}")

    print(f"Total decode time                      : {total_t_decode:.4f} s")
    print(f"Total beam time                        : {total_t_beam  :.4f} s")
    print(f"Total rand time                        : {total_t_rand:.4f}   s")

    print(f"Averge beam search length              : {avg_bs_len:.2f}")
    print(f"Average beam search number of nodes in the path:{avg_bs_nodes:.2f}")
    print(f"Average decode length                  :{avg_dec_len}")
    print(f"Average decode number of nodes in the path:{avg_dec_nodes:.2f}")
    print(f"Average random-0.5 path length              : {avg_rand_len:.2f}")
    print(f"Average random-0.5 nodes in path            : {avg_rand_nodes:.2f}")

    # return v_true, v_pred, ratio_list
    metrics = dict(
        better_rate=better_rate,
        rmse=rmse,
        overall_gap=overall_gap,
        overall_gap_beam_lp=overall_gap_beam_lp,
        negative_rate=neg_rate,

        total_t_decode=total_t_decode,
        total_t_beam=total_t_beam,
        total_t_rand=total_t_rand,

        avg_bs_len=avg_bs_len,
        avg_bs_nodes=avg_bs_nodes,
        avg_dec_len=avg_dec_len,
        avg_dec_nodes=avg_dec_nodes,
        avg_rand_len=avg_rand_len,
        avg_rand_nodes=avg_rand_nodes,

        better_rate_rand=better_rate_rand,
        avg_ratio_rand=avg_ratio_rand,
        overall_gap_rand=overall_gap_rand,
        rmse_rand=rmse_rand,
        negative_rate_rand=neg_rate_rand,
    )
    return metrics


# main
def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_data = torch.load(args.graph_file, weights_only=False)
    test_loader = DataLoader(test_data, batch_size=len(test_data),
                             shuffle=False)

    for d in test_data: d.weight = d.edge_attr[:, 0]
    test_graphs = [
        to_networkx(d, to_undirected=False, edge_attrs=["weight"])
        for d in test_data
    ]

    input_dim = test_data[0].x.size(1)
    common_kwargs = dict(
        num_layers=3, hidden1=64, hidden2=64,
        in_dim=input_dim, K_internal=3,
        edge_in_dim=2, pos_dim=2, dropout=0.2,
    )
    model = build_model(**common_kwargs).to(device)

    model.load_state_dict(torch.load(args.model_file, map_location=device))

    decoded_info = []
    t0 = time.time()
    model.eval()
    with torch.no_grad():
        global_graph_index = 0
        for batch_data in test_loader:
            batch_data = batch_data.to(device)
            retdict = model(batch_data)
            edge_probs_batch = retdict["edge_probs"]
            batch_vec = batch_data.batch
            edge_src, edge_dst = batch_data.edge_index
            num_graphs_in_batch = batch_vec.max().item() + 1

            for g in range(num_graphs_in_batch):
                node_mask = (batch_vec == g)
                node_idx = node_mask.nonzero(as_tuple=True)[0]

                id_map = -torch.ones(batch_data.num_nodes,
                                     dtype=torch.long, device=device)
                id_map[node_idx] = torch.arange(node_idx.size(0),
                                                device=device)

                src_local = batch_data.source[g].item()
                snk_local = batch_data.sink[g].item()

                edge_mask = node_mask[edge_src] & node_mask[edge_dst]
                edge_idx = edge_mask.nonzero(as_tuple=True)[0]

                edge_index_g = batch_data.edge_index[:, edge_idx]
                edge_index_g_mapped = id_map[edge_index_g].t().cpu().tolist()
                edge_probs_g = edge_probs_batch[edge_idx].cpu()
                edge_probs_rand = torch.full_like(edge_probs_g, 0.5)

                G_g = test_graphs[global_graph_index]

                t_decode = time.perf_counter()

                path_g = decode_sampled_path_edge(
                    G_g, edge_index_g_mapped, edge_probs_g,
                    src_local, snk_local, num_samples=50, seed=123
                )

                t_decode = time.perf_counter() - t_decode

                t_rand = time.perf_counter()
                path_rand = decode_sampled_path_edge(
                    G_g, edge_index_g_mapped, edge_probs_rand,
                    src_local, snk_local, num_samples=50
                )
                t_rand = time.perf_counter() - t_rand

                t_beam = time.perf_counter()
                _, _ = beam_search(G_g, src_local, snk_local,
                                   beam_width=200)
                t_beam = time.perf_counter() - t_beam

                decoded_info.append({
                    "graph_index": global_graph_index,
                    "decoded_path": path_g,
                    "rand_path": path_rand,
                    "source": src_local,
                    "sink": snk_local,
                    "time_beam": t_beam,
                    "time_decode": t_decode,
                    "time_rand": t_rand,
                })
                global_graph_index += 1

    compare_decoded_paths(decoded_info, test_graphs,
                          t0, 0, 0, model_label='ESPP-NNAA')


if __name__ == "__main__":
    random.seed(42)
    main()
