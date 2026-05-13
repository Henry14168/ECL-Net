import re
import pickle
import os
import torch
import numpy as np
import traceback
from torch_geometric.data import Data
class PairData(Data):
    def __inc__(self, key, value, *args):

        if key == 'edge_index_s': return self['x_s'].size(0)
        if key == 'mut_idx_s': return self['x_s'].size(0)
        if key == 'env_idx_s': return self['x_s'].size(0)

        if key == 'edge_index_t': return self['x_t'].size(0)
        if key == 'mut_idx_t': return self['x_t'].size(0)
        if key == 'env_idx_t': return self['x_t'].size(0)
        return super().__inc__(key, value, *args)

def read_gpickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def _parse_line(raw: str):
    s = raw.strip()
    if not s: return None, None
    parts = s.split()

    if len(parts) == 5:
        base = f"{parts[0]}_{parts[2]}{parts[1]}{parts[3]}"
        try:
            label = float(parts[4])
        except:
            label = None
        return base, label

    if re.match(r"^[0-9A-Za-z]+_[A-Z]\d+[A-Z]$", s):
        return s, None

    return s.replace(" ", ""), None

def find_graph_file(base_dir, base_name, suffix_type="wt"):
    target = f"{base_name}_{suffix_type}.pkl"
    path = os.path.join(base_dir, target)
    if os.path.exists(path): return path

    clean_name = base_name.replace("_relaxed", "")
    target_clean = f"{clean_name}_{suffix_type}.pkl"
    path_clean = os.path.join(base_dir, target_clean)
    if os.path.exists(path_clean): return path_clean
    return None

def process_single_graph(G):
    nodes = list(G.nodes(data=True))
    nodes.sort(key=lambda x: x[0])
    num_nodes = len(nodes)

    try:
        node_feats = [d['node_s'] for _, d in nodes]
        x = torch.from_numpy(np.stack(node_feats)).float()
    except KeyError:
        x = torch.zeros((num_nodes, 1), dtype=torch.float)

    try:
        node_pos = [d['pos'] for _, d in nodes]
        pos = torch.from_numpy(np.stack(node_pos)).float()
    except:
        pos = torch.zeros((num_nodes, 3), dtype=torch.float)

    if G.number_of_edges() > 0:
        edges = list(G.edges(data=True))
        u_list, v_list, edge_feats_list = [], [], []

        for u, v, d in edges:
            u_list.append(u)
            v_list.append(v)
            if 'edge_s' in d:
                edge_feats_list.append(d['edge_s'])

        edge_index = torch.tensor([u_list, v_list], dtype=torch.long)
        if edge_feats_list:
            edge_attr = torch.from_numpy(np.stack(edge_feats_list)).float()
        else:
            edge_attr = None
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = None

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos, num_nodes=num_nodes)

    mut_idx_val = None
    if 'mut_idx' in G.graph:
        mut_idx_val = G.graph['mut_idx']
    elif 'mut_pos' in G.graph:
        mut_idx_val = G.graph['mut_pos']

    if mut_idx_val is not None:
        data.mut_idx = torch.tensor([mut_idx_val], dtype=torch.long)
    else:
        data.is_invalid = True
        return data

    FIXED_ENV_LEN = 32

    env_list = G.graph.get('env_idx', [])
    if not isinstance(env_list, list): env_list = list(env_list)

    score_list = G.graph.get('env_scores', [1.0] * len(env_list))
    if not isinstance(score_list, list): score_list = list(score_list)

    if len(score_list) != len(env_list):
        score_list = [1.0] * len(env_list)

    current_len = len(env_list)

    if current_len > FIXED_ENV_LEN:
        env_list = env_list[:FIXED_ENV_LEN]
        score_list = score_list[:FIXED_ENV_LEN]

    elif current_len < FIXED_ENV_LEN:
        pad_len = FIXED_ENV_LEN - current_len
        env_list = env_list + [0] * pad_len
        score_list = score_list + [0.0] * pad_len

    data.env_idx = torch.tensor(env_list, dtype=torch.long).unsqueeze(0)
    data.env_weight = torch.tensor(score_list, dtype=torch.float).unsqueeze(0)

    return data

def load_dataset(graph_dir, split="train", labeled=True, include_reverse=True):
    cache_path = os.path.join(graph_dir, f"cached_{split}.pt")
    if os.path.exists(cache_path):
        print(f"[Info] Found cache for {split}, loading directly...")
        try:

            data_list = torch.load(cache_path, weights_only=False)
            print(f"{split.upper()} LOADED from CACHE: {len(data_list)} samples")
            return data_list
        except Exception as e:
            print(f"[Warn] Cache corrupted, reloading: {e}")

    data_list = []
    file_candidates = [
        os.path.join("data", "datasets", f"{split}.txt"),
        os.path.join("data", f"{split}.txt")
    ]
    name_file = None
    for fpath in file_candidates:
        if os.path.exists(fpath):
            name_file = fpath
            break

    if not name_file:
        print(f"[Error] Not found list file for {split}")
        return []

    dataset_dir = os.path.join(graph_dir, split)
    print(f"[Info] Loading {split} from: {name_file}")

    mode_str = "FORWARD + REVERSE" if (labeled and include_reverse) else "FORWARD ONLY"
    print(f"[Info] Processing mode: {mode_str} (Padding to 32)")

    with open(name_file, 'r') as f:
        lines = f.readlines()

    skip_reason = {"file_missing": 0, "invalid_graph": 0, "error": 0}

    for i, line in enumerate(lines):
        base, label_from_txt = _parse_line(line)
        if not base: continue

        path_wt = find_graph_file(dataset_dir, base, "wt")
        path_mut = find_graph_file(dataset_dir, base, "mut")

        if not path_wt or not path_mut:
            skip_reason["file_missing"] += 1
            continue

        try:
            G_wt = read_gpickle(path_wt)
            G_mut = read_gpickle(path_mut)

            data_wt = process_single_graph(G_wt)
            data_mut = process_single_graph(G_mut)

            if hasattr(data_wt, 'is_invalid') or hasattr(data_mut, 'is_invalid'):
                skip_reason["invalid_graph"] += 1
                continue

            label_tensor = torch.tensor([0.0], dtype=torch.float)
            if labeled:
                if label_from_txt is not None:
                    label_tensor = torch.tensor([float(label_from_txt)], dtype=torch.float)
                elif 'y' in G_wt.graph:
                    label_tensor = torch.tensor([float(G_wt.graph['y'])], dtype=torch.float)

            wt_count = torch.tensor([data_wt.num_nodes], dtype=torch.long)
            mut_count = torch.tensor([data_mut.num_nodes], dtype=torch.long)

            data_direct = PairData(
                x_s=data_wt.x, pos_s=data_wt.pos,
                edge_index_s=data_wt.edge_index, edge_attr_s=data_wt.edge_attr,
                mut_idx_s=data_wt.mut_idx, env_idx_s=data_wt.env_idx,
                env_weight_s=data_wt.env_weight,

                x_t=data_mut.x, pos_t=data_mut.pos,
                edge_index_t=data_mut.edge_index, edge_attr_t=data_mut.edge_attr,
                mut_idx_t=data_mut.mut_idx, env_idx_t=data_mut.env_idx,
                env_weight_t=data_mut.env_weight,

                y=label_tensor, name=base, wt_count=wt_count, mut_count=mut_count
            )
            data_list.append(data_direct)

            if include_reverse and labeled:
                rev_label = torch.tensor([1.0 - label_tensor.item()], dtype=torch.float)
                data_reverse = PairData(

                    x_s=data_mut.x, pos_s=data_mut.pos,
                    edge_index_s=data_mut.edge_index, edge_attr_s=data_mut.edge_attr,
                    mut_idx_s=data_mut.mut_idx, env_idx_s=data_mut.env_idx,
                    env_weight_s=data_mut.env_weight,

                    x_t=data_wt.x, pos_t=data_wt.pos,
                    edge_index_t=data_wt.edge_index, edge_attr_t=data_wt.edge_attr,
                    mut_idx_t=data_wt.mut_idx, env_idx_t=data_wt.env_idx,
                    env_weight_t=data_wt.env_weight,

                    y=rev_label, name=f"{base}_rev", wt_count=mut_count, mut_count=wt_count
                )
                data_list.append(data_reverse)

            if len(data_list) > 0 and len(data_list) % 1000 == 0:
                print(f"    ... Processed {len(data_list)} pairs ...")

        except Exception as e:
            print(f"Error processing {base}: {e}")
            traceback.print_exc()
            skip_reason["error"] += 1

    print(f'{split.upper()} PROCESSED: {len(data_list)} samples')
    if len(data_list) == 0:
        print(f"[Debug] Skip Details: {skip_reason}")
    else:

        print(f"[Info] Saving cache to {cache_path}...")
        try:
            torch.save(data_list, cache_path)
        except Exception as e:
            print(f" [Warn] Failed to save cache (likely Disk Full). Continuing without caching... Error: {e}")

    return data_list