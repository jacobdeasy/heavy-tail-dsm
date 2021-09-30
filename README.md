Instructions for running experiments:

1. Prepare experiment settings:
    
    `configs/<config_file>.yml`
2. Training:

    `python main.py --config <config_file>.yml --doc <save_dir> --beta <experiment_beta>`
3. Sampling (adjust `fid` parameter in `configs/<config_file>.yml` if need be):

    `python main.py --config <config_file>.yml --doc <save_dir> --beta <experiment_beta> --sample -i <save_dir>`
4. Metrics (after preparing metric comparison data in single-level `<compare_dir>`):

    Windows: `.\metrics.ps1 .\exp\image_samples\<save_dir>\ .\exp\datasets\<compare_dir>\`

    Linux - `./metrics.sh ./exp/image_samples/<save_dir>/ ./exp/datasets/<compare_dir>/`
