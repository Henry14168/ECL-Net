import torch
import esm
from Bio import SeqIO
import os

device = torch.device("cuda")
model_name = "esm2_t36_3B_UR50D"

model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
model.eval()
model.to(device)

batch_converter = alphabet.get_batch_converter()

input_root = "data/fasta"
output_root = "ESM2"
os.makedirs(output_root, exist_ok=True)

position_log_path = "mutation_positions.txt"
print(f"[INFO] Mutation positions will be saved to: {position_log_path}")
TOP_K = 32

def get_real_mutation_pos(mut_filename, mut_seq, current_subdir):
    try:
        pdb_id = mut_filename.split('_')[0]
        wt_filename = f"{pdb_id}_relaxed.fasta"
        wt_path = os.path.join(input_root, current_subdir, wt_filename)

        if mut_filename == wt_filename or not os.path.exists(wt_path):
            return None

        wt_records = list(SeqIO.parse(wt_path, "fasta"))
        if not wt_records: return None
        wt_seq = str(wt_records[0].seq)

        if len(wt_seq) != len(mut_seq): return None

        for i, (w, m) in enumerate(zip(wt_seq, mut_seq)):
            if w != m: return i
    except:
        pass
    return None

with open(position_log_path, "w") as log_file:
    log_file.write("ID Index_0_based\n")

    for sub in sorted(os.listdir(input_root)):
        in_subdir = os.path.join(input_root, sub)
        if not os.path.isdir(in_subdir): continue

        out_subdir = os.path.join(output_root, sub)
        os.makedirs(out_subdir, exist_ok=True)

        print(f"\n[INFO] Processing directory: {sub}")
        files = sorted([f for f in os.listdir(in_subdir) if f.lower().endswith((".fasta", ".fa"))])

        for fname in files:
            safe_id = os.path.splitext(fname)[0]
            out_path = os.path.join(out_subdir, f"{safe_id}.pt")

            if os.path.exists(out_path):
                pass

            records = list(SeqIO.parse(os.path.join(in_subdir, fname), "fasta"))
            if not records: continue
            seq_id = records[0].id
            sequence = str(records[0].seq)

            if len(sequence) > 1022: sequence = sequence[:1022]

            try:
                real_mut_idx = get_real_mutation_pos(fname, sequence, sub)

                if real_mut_idx is not None:
                    clean_id = safe_id.replace("_relaxed", "")
                    log_file.write(f"{clean_id} {real_mut_idx}\n")
                    log_file.flush()

                if os.path.exists(out_path):
                    continue

                batch_labels, batch_strs, batch_tokens = batch_converter([(seq_id, sequence)])
                batch_tokens = batch_tokens.to(device)

                with torch.no_grad():
                    results = model(batch_tokens, repr_layers=[36], need_head_weights=True, return_contacts=False)

                token_rep = results["representations"][36]
                embedding = token_rep[0, 1: len(sequence) + 1].cpu()

                final_indices = None
                final_scores = None

                if real_mut_idx is not None:
                    target_token_idx = real_mut_idx + 1

                    if target_token_idx <= len(sequence):
                        attentions = results["attentions"][0, -1, :, :, :]
                        avg_attn = attentions.mean(dim=0)
                        scores = avg_attn[target_token_idx, :]

                        valid_mask = torch.zeros_like(scores, dtype=torch.bool)
                        valid_mask[1: len(sequence) + 1] = True
                        valid_mask[target_token_idx] = False

                        scores[~valid_mask] = float('-inf')

                        k = min(TOP_K, valid_mask.sum().item())
                        topk_scores, topk_token_idx = torch.topk(scores, k=k)

                        final_indices = (topk_token_idx - 1).cpu()
                        final_scores = topk_scores.cpu()

                data_payload = {
                    "embedding": embedding,
                    "top32_idx": final_indices,
                    "top32_scores": final_scores
                }
                torch.save(data_payload, out_path)

                status = f"Indexed (Pos {real_mut_idx})" if final_indices is not None else "EmbOnly"
                print(f"  [Done] {safe_id} -> {status}")

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"  [OOM] {safe_id}")
                    torch.cuda.empty_cache()
                else:
                    print(f"  [Error] {safe_id}: {e}")

print("\n[INFO] All tasks finished.")