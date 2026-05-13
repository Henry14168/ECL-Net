import argparse
import os
import warnings
import networkx as nx
import pickle
import torch
import numpy as np
import re
import traceback
from Bio.PDB import PDBParser, is_aa
from scipy.spatial import cKDTree
class MockAtom:
    def __init__(self, coords):
        self._coords = np.array(coords, dtype=np.float32)

    def get_coord(self):
        return self._coords

class MockResidue:
    def __init__(self, res_seq, res_icode, res_name, coords_dict):
        self.id = (' ', res_seq, res_icode)
        self.resname = res_name
        self.coords_dict = coords_dict
    def get_resname(self):
        return self.resname
    def __contains__(self, item):
        return item in self.coords_dict

    def __getitem__(self, item):
        return MockAtom(self.coords_dict[item])
def parse_pdb_force_mode(path, target_chain='A'):
    res_list = []

    curr_seq = None
    curr_icode = None
    curr_name = None
    curr_coords = {}

    seen_ids = set()

    with open(path, 'r', errors='ignore') as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')):
                try:
                    if len(line) < 54: continue

                    chain_id = line[21].strip()
                    if target_chain and chain_id and chain_id != target_chain:
                        continue

                    atom = line[12:16].strip()
                    resn = line[17:20].strip()

                    res_seq_str = line[22:26].strip()
                    icode = line[26].strip()

                    try:
                        seq = int(res_seq_str)
                    except:
                        m = re.search(r'(-?\d+)', line[22:27])
                        if m:
                            seq = int(m.group(1))
                        else:
                            continue
                        if 'X' in line[22:27]: seq = 0

                    if (seq != curr_seq) or (icode != curr_icode):
                        if curr_seq is not None and 'CA' in curr_coords:
                            unique_id = (curr_seq, curr_icode)
                            if unique_id not in seen_ids:
                                res_list.append(MockResidue(curr_seq, curr_icode, curr_name, curr_coords))
                                seen_ids.add(unique_id)

                        curr_seq = seq
                        curr_icode = icode
                        curr_name = resn
                        curr_coords = {}

                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    curr_coords[atom] = [x, y, z]

                except:
                    continue

        if curr_seq is not None and 'CA' in curr_coords:
            unique_id = (curr_seq, curr_icode)
            if unique_id not in seen_ids:
                res_list.append(MockResidue(curr_seq, curr_icode, curr_name, curr_coords))

    return res_list

def get_coords(res, atom_name):
    try:
        if atom_name in res:
            return torch.tensor(res[atom_name].get_coord(), dtype=torch.float32)
    except:
        pass
    return None

def rbf_expansion(distance, K=16, d_min=0.0, d_max=20.0):
    centers = torch.linspace(d_min, d_max, K)
    sigma = centers[1] - centers[0]
    rbf_vector = torch.exp(-((distance - centers) / sigma) ** 2)
    return rbf_vector.tolist()

def get_node_feature(nodes_list, esm_embedding):
    features = []
    if isinstance(esm_embedding, torch.Tensor) and esm_embedding.dim() == 1:
        esm_embedding = esm_embedding.unsqueeze(0)

    length = min(len(nodes_list), esm_embedding.shape[0])
    for i in range(length):
        features.append(esm_embedding[i].numpy())
    return features

def load_esm_data_smart(esm_dir, base_id):
    candidates = [f"{base_id}_relaxed.pt", f"{base_id}.pt"]
    for fname in candidates:
        path = os.path.join(esm_dir, fname)
        if os.path.exists(path):
            try:
                data = torch.load(path)
                if isinstance(data, dict):
                    return data['embedding'], data.get('top32_idx', None), data.get('top32_scores', None), path
                elif torch.is_tensor(data):
                    return data, None, None, path
            except:
                pass
    return None, None, None, None

def make_graph(record, pdb_root_dir, esm_dir, out_dir, is_wt=True, split="train",
               top_k=30, label_threshold=0.0):
    parts = record.strip().split()
    pdb_name, mut_pos, wt, mut_field, label_val = "", "", "", "", None

    if len(parts) == 5:
        pdb_name, mut_pos, wt, mut_field = parts[0], parts[1], parts[2], parts[3]
        try:
            label_val = float(parts[4])
        except:
            label_val = None
    elif len(parts) == 2:
        s, val_str = parts[0], parts[1]
        try:
            label_val = 1.0 if float(val_str) > label_threshold else 0.0
        except:
            pass
        if '_' in s:
            pdb_name, mut_info = s.split('_', 1)
            m = re.match(r"([A-Z])(\d+)([A-Z]?)", mut_info)
            if m: wt, mut_pos, mut_field = m.groups()
    elif len(parts) == 1:
        if '_' in parts[0]:
            pdb_name, mut_info = parts[0].split('_', 1)
            m = re.match(r"([A-Z])(\d+)([A-Z]?)", mut_info)
            if m: wt, mut_pos, mut_field = m.groups()
    else:
        return

    if not mut_field: mut_field = ""

    def resolve_pdb_path(base_dir, pdb_name, is_wt):
        tag = f"{wt}{mut_pos}{mut_field}"
        candidates = []
        if is_wt:
            base = os.path.join(base_dir, f"{pdb_name}_relaxed")
            candidates = [base + ".pdb", base]
        else:
            candidates.append(os.path.join(base_dir, f"{pdb_name}_{tag}_relaxed.pdb"))
            candidates.append(os.path.join(base_dir, f"{pdb_name}_{tag}_relaxed_{tag}_relaxed.pdb"))
            candidates.append(os.path.join(base_dir, f"{pdb_name}_{tag}.pdb"))
        for path in candidates:
            if os.path.exists(path): return path
        return candidates[0]

    os.makedirs(os.path.join(out_dir, split), exist_ok=True)
    try:
        mut_pos_int = int(mut_pos)
    except:
        return

    if is_wt:
        out_path = f"{out_dir}/{split}/{pdb_name}_{wt}{mut_pos}{mut_field}_wt.pkl"
        current_seq_id = pdb_name
    else:
        out_path = f"{out_dir}/{split}/{pdb_name}_{wt}{mut_pos}{mut_field}_mut.pkl"
        current_seq_id = f"{pdb_name}_{wt}{mut_pos}{mut_field}"

    base_dir = os.path.join(pdb_root_dir, split)
    pdb_path = resolve_pdb_path(base_dir, pdb_name, is_wt)

    if not os.path.exists(pdb_path) or os.path.getsize(pdb_path) == 0:
        return

    res_list = []

    try:
        p = PDBParser(PERMISSIVE=1, QUIET=True)
        structure = p.get_structure(pdb_name, pdb_path)
        model0 = structure[0]

        chain_id = pdb_name[-1] if len(pdb_name) > 4 else 'A'
        chain = None
        if chain_id in model0:
            chain = model0[chain_id]
        else:
            all = list(model0.get_chains())
            if all:
                chain = all[0]
            else:
                raise ValueError("No chain")

        for res in chain:
            if is_aa(res, standard=True) and "CA" in res:
                res_list.append(res)

        if not res_list: raise ValueError("No CA")

    except Exception:
        chain_id = pdb_name[-1] if len(pdb_name) > 4 else 'A'
        res_list = parse_pdb_force_mode(pdb_path, chain_id)
        if not res_list: return
    def sort_key(res):
        return (res.id[1], res.id[2])

    res_list.sort(key=sort_key)
    G = nx.Graph()
    if label_val is not None: G.graph['y'] = label_val

    coords_list = []
    found_mut_idx = None
    for i, res in enumerate(res_list):
        center = get_coords(res, 'CA')
        if center is None: continue
        G.add_node(i, name=res.get_resname(), pos=center.numpy())
        coords_list.append(center.numpy())
        if res.id[1] == mut_pos_int and res.id[2].strip() == '':
            found_mut_idx = i

    if found_mut_idx is not None:
        G.graph['mut_idx'] = found_mut_idx
    else:
        return
    nodes_list = list(G.nodes)

    if len(coords_list) > 1:
        coords_array = np.array(coords_list)
        kdtree = cKDTree(coords_array)
        k = min(top_k + 1, len(coords_array))
        dists, indices = kdtree.query(coords_array, k=k)

        if k == 1:
            dists, indices = [dists], [indices]

        for i, (nbr_dists, nbr_indices) in enumerate(zip(dists, indices)):
            u = nodes_list[i]

            if not isinstance(nbr_dists, (list, np.ndarray)): nbr_dists = [nbr_dists]
            if not isinstance(nbr_indices, (list, np.ndarray)): nbr_indices = [nbr_indices]

            for d, j in zip(nbr_dists, nbr_indices):
                if i == j: continue
                v = nodes_list[j]
                edge_s = rbf_expansion(torch.tensor(d), K=16, d_max=20.0)
                G.add_edge(u, v, edge_s=edge_s)
    esm_base = os.path.join(esm_dir, split)
    target_mut_id = f"{pdb_name}_{wt}{mut_pos}{mut_field}"

    try:
        my_emb, my_idx, my_scores, _ = load_esm_data_smart(esm_base, current_seq_id)
        final_idx, final_scores = None, None

        if is_wt:
            _, borrowed_idx, borrowed_scores, _ = load_esm_data_smart(esm_base, target_mut_id)
            final_idx = borrowed_idx
            final_scores = borrowed_scores
        else:
            final_idx = my_idx
            final_scores = my_scores

        if my_emb is not None:
            feats = get_node_feature(nodes_list, my_emb)
            for i, n_id in enumerate(nodes_list):
                if i < len(feats): G.nodes[n_id]['node_s'] = feats[i]

        if final_idx is not None:
            if isinstance(final_idx, torch.Tensor) and final_idx.dim() == 0:
                final_idx = final_idx.unsqueeze(0)

            env_idxs = []
            for idx in final_idx:
                i_val = int(idx)
                if i_val < len(nodes_list): env_idxs.append(i_val)
            G.graph['env_idx'] = env_idxs
        else:
            if 'mut_idx' in G.graph:
                m_i = G.graph['mut_idx']
                _, k_idxs = kdtree.query(coords_array[m_i], k=33)
                env_idxs = [int(x) for x in k_idxs if x != m_i][:32]
                G.graph['env_idx'] = env_idxs

        if final_scores is not None:
            G.graph['env_scores'] = final_scores.tolist()

    except Exception:
        pass
    mapping = {node_id: i for i, node_id in enumerate(nodes_list)}
    nx.relabel_nodes(G, mapping, copy=False)

    with open(out_path, 'wb') as f:
        pickle.dump(G, f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--feature_path', type=str, default=None)
    parser.add_argument('--esm_dir', type=str, default='ESM2')
    parser.add_argument('--out_dir', type=str, default='data/graphs')
    parser.add_argument('--split', type=str, default="S2648")
    parser.add_argument('--pdb_root', type=str, default='data/relax')
    parser.add_argument('--top_k', type=int, default=30)
    parser.add_argument('--label_threshold', type=float, default=0.0)

    args = parser.parse_args()
    if args.feature_path is None: args.feature_path = args.data_path

    warnings.filterwarnings('ignore')

    print(f"Start generating GRAPHS for {args.split}...")

    count = 0
    with open(args.data_path, 'r') as f:
        for line in f:
            name = line.strip()
            if not name: continue
            try:
                make_graph(name, args.pdb_root, args.esm_dir, args.out_dir, True, args.split, args.top_k,
                           args.label_threshold)
                make_graph(name, args.pdb_root, args.esm_dir, args.out_dir, False, args.split, args.top_k,
                           args.label_threshold)
                count += 1
                if count % 100 == 0: print(f"Processed {count}...")
            except Exception as e:
                print(f" Critical Error on {name}: {e}")
                traceback.print_exc()

    print(f"Done. {count} proteins processed.")


if __name__ == "__main__":
    main()