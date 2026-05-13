import os
import sys
import subprocess
import multiprocessing
from datetime import datetime
ROSETTA_RELAX_BIN = os.path.expanduser(
    'relax.mpi.linuxgccrelease'
)
DATASETS = ['TG', 'S2648']

def wild_relax(start_struct, pdb_chain):
    input_pdb_path = os.path.abspath('data/pdbs')
    output_base_path = os.path.abspath('data/relax_wt')

    subdir = os.path.dirname(start_struct)
    output_dir = os.path.join(output_base_path, subdir) if subdir else output_base_path
    os.makedirs(output_dir, exist_ok=True)

    start_struct_path = os.path.join(input_pdb_path, start_struct)

    args = [
        ROSETTA_RELAX_BIN,
        '-in:file:s', start_struct_path,
        '-in:file:fullatom',
        '-relax:constrain_relax_to_start_coords',
        '-out:no_nstruct_label',
        '-relax:ramp_constraints', 'false',
        '-default_max_cycles', '200',
        '-out:file:scorefile', f"{pdb_chain}_relaxed.sc",
        '-out:suffix', '_relaxed'
    ]

    log_path = os.path.join(output_dir, f'rosetta_{pdb_chain}.out')

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
    base_pdb_path = os.path.abspath('data/pdbs')

    max_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', multiprocessing.cpu_count()))
    tasks = []

    print(f"[INFO] Initializing wild-type relaxation for datasets: {DATASETS}")

    for dataset in DATASETS:
        target_path = os.path.join(base_pdb_path, dataset)
        if not os.path.exists(target_path):
            print(f"[WARNING] Directory missing: {target_path}. Skipping.")
            continue

        for file_name in os.listdir(target_path):
            if file_name.endswith('.pdb'):
                rel_path = os.path.join(dataset, file_name)
                pdb_chain = os.path.splitext(file_name)[0]
                tasks.append((rel_path, pdb_chain))

    if not tasks:
        print("[INFO] No PDB files found. Exiting.")
        sys.exit(0)

    print(f"[INFO] Dispatched {len(tasks)} tasks across {max_cpus} cores.")

    with multiprocessing.Pool(processes=min(max_cpus, len(tasks))) as pool:
        pool.starmap(wild_relax, tasks)

    print(f"[INFO] Pipeline completed in {datetime.now() - start_time}")

if __name__ == '__main__':
    main()