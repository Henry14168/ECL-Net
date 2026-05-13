import logging
from pathlib import Path
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from tqdm import tqdm

AA_MAP = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    'MSE': 'M', 'HSD': 'H', 'HSE': 'H', 'HSP': 'H', 'HIS_D': 'H'
}

def extract_sequence_force(pdb_path: Path) -> str | None:
    residues = []
    seen = set()

    try:
        with open(pdb_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if not (line.startswith('ATOM') and 'CA ' in line[12:16]):
                    continue

                try:
                    res_name = line[17:20].strip()
                    chain_id = line[21].strip()
                    res_seq = int(line[22:26].strip())
                    icode = line[26].strip()

                    unique_id = (chain_id, res_seq, icode)

                    if unique_id not in seen:
                        seen.add(unique_id)
                        aa = AA_MAP.get(res_name.upper(), 'X')
                        residues.append((unique_id, aa, chain_id))
                except (ValueError, IndexError):
                    continue

    except IOError as e:
        logging.debug(f"File read error for {pdb_path}: {e}")
        return None

    if not residues:
        return None
    residues.sort(key=lambda x: x[0])

    chains = {}
    for _, aa, chain_id in residues:
        chains.setdefault(chain_id, []).append(aa)

    return ":".join("".join(chains[c]) for c in sorted(chains.keys()))

def extract_sequence_smart(pdb_path: Path) -> str | None:
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure('struct', str(pdb_path))
        sequences = []

        for model in structure:
            for chain in model:
                chain_seq = []
                res_list = sorted(list(chain.get_residues()), key=lambda x: (x.id[1], x.id[2]))

                for res in res_list:
                    res_name = res.get_resname()
                    if (res_name in AA_MAP or res_name == 'MSE') and 'CA' in res:
                        name_to_convert = 'MET' if res_name == 'MSE' else res_name
                        try:
                            chain_seq.append(seq1(name_to_convert))
                        except KeyError:
                            continue
                if chain_seq:
                    sequences.append("".join(chain_seq))
            break

        if sequences:
            return ":".join(sequences)

    except Exception as e:
        logging.debug(f"Biopython failed for {pdb_path}: {e}")

    return extract_sequence_force(pdb_path)

def main():
    input_root = Path("data/relax")
    output_root = Path("data/fasta")

    if not input_root.exists():
        logging.error(f"Input directory does not exist: {input_root}")
        return

    pdb_files = list(input_root.rglob("*.pdb"))

    if not pdb_files:
        logging.warning(f"No .pdb files found in {input_root}")
        return

    logging.info(f"Found {len(pdb_files)} PDB files. Starting extraction...")

    success_count = 0

    for pdb_path in tqdm(pdb_files, desc="Processing PDBs"):
        rel_path = pdb_path.relative_to(input_root)
        output_dir = output_root / rel_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        sequence = extract_sequence_smart(pdb_path)

        if sequence:
            fasta_path = output_dir / f"{pdb_path.stem}.fasta"
            with open(fasta_path, 'w', encoding='utf-8') as f:
                f.write(f">{pdb_path.stem}\n{sequence}\n")
            success_count += 1

    logging.info("Processing complete.")
    logging.info(f"Successfully extracted: {success_count}/{len(pdb_files)}")
    logging.info(f"Results saved to: {output_root}")


if __name__ == "__main__":
    main()