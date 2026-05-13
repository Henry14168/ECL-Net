import os
import sys
import subprocess
import multiprocessing
from datetime import datetime
ROSETTA_RELAX_BIN = os.path.expanduser(
    'relax.mpi.linuxgccrelease'
)
DATASETS = ['TG', 'S2648']

def wild_mutate(start_struct, pdb_chain, variant_resfile, variant):
    output_dir = os.path.dirname(os.path.abspath(variant_resfile))
    start_struct_path = os.path.abspath(start_struct)

    args = [
        ROSETTA_RELAX_BIN,
        '-in:file:s', start_struct_path,
        '-in:file:fullatom',
        '-relax:constrain_relax_to_start_coords',
        '-out:no_nstruct_label',
        '-relax:ramp_constraints', 'false',
        '-relax:respect_resfile',
        '-packing:resfile', os.path.abspath(variant_resfile),
        '-default_max_cycles', '200',
        '-out:file:scorefile', f"{pdb_chain[:5]}_{variant}_relaxed.sc",
        '-out:suffix', f"_{variant}_relaxed"
    ]

    log_path = os.path.join(output_dir, f'rosetta_{pdb_chain[:5]}_{variant}.out')

    with open(log_path, 'w') as outfile:
        process = subprocess.Popen(
            args,
            stdout=outfile,
            stderr=subprocess.STDOUT,
            cwd=output_dir
        )
        process.wait()

def main():
    start_time = datetime.now()
    base_path = os.path.abspath('data')

    # Priority: SLURM environment variable > local CPU count
    max_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', multiprocessing.cpu_count()))
    tasks = []

    print(f"[INFO] Initializing mutation pipeline for datasets: {DATASETS}")

    for dataset in DATASETS:
        input_dir = os.path.join(base_path, 'relax_wt', dataset)
        output_dir = os.path.join(base_path, 'relax_mut', dataset)
        variant_list = os.path.join(base_path, f'{dataset}.txt')

        if not os.path.isfile(variant_list):
            print(f"[WARNING] Variant list missing: {variant_list}. Skipping.")
            continue

        os.makedirs(output_dir, exist_ok=True)

        with open(variant_list, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    pdb_chain, pos, wt_aa, mut_aa = parts[:4]
                    variant = f"{wt_aa}{pos}{mut_aa}"

                    start_struct = os.path.join(input_dir, f"{pdb_chain[:5]}_relaxed.pdb")
                    if not os.path.exists(start_struct):
                        continue

                    variant_resfile = os.path.join(output_dir, f"{pdb_chain}_{variant}.resfile")
                    with open(variant_resfile, 'w') as res_f:
                        res_f.write(f"NATAA\nstart\n{pos} {pdb_chain[4]} PIKAA {mut_aa}\n")

                    tasks.append((start_struct, pdb_chain, variant_resfile, variant))

    if not tasks:
        print("[INFO] No valid variants found. Exiting.")
        sys.exit(0)

    print(f"[INFO] Dispatched {len(tasks)} tasks across {max_cpus} cores.")

    with multiprocessing.Pool(processes=min(max_cpus, len(tasks))) as pool:
        pool.starmap(wild_mutate, tasks)

    print(f"[INFO] Pipeline completed in {datetime.now() - start_time}")

if __name__ == '__main__':
    main()