#!/bin/bash  
#SBATCH --job-name=YOLOX_train
#SBATCH --output="/leonardo/home/userexternal/fmorandi/MOTR_domain_gap/exps/slurm_logs/%x_%A_%a.out"
#SBATCH --error="/leonardo/home/userexternal/fmorandi/MOTR_domain_gap/exps/slurm_logs/%x_%A_%a.err"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --mem=160GB
#SBATCH --cpus-per-gpu=8
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrB_FeeCO
#SBATCH --time=06:00:00
#SBATCH --qos=boost_qos_lprod
##SBATCH --qos=boost_qos_dbg


source /leonardo/home/userexternal/fmorandi/envs/motr/bin/activate
pwd

# Controlla che Conda sia attivato correttamente
which python  # Dovrebbe stampare il percorso dell'interprete Python di bytetrack
python --version  # Controlla che la versione di Python sia quella prevista

# Controlla la presenza delle GPU
echo ">>> GPU disponibili:"
nvidia-smi  # Deve mostrare le GPU disponibili

# Controlla se PyTorch rileva le GPU
echo ">>> PyTorch CUDA availability:"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Controlla la versione di CUDA usata da PyTorch
echo ">>> PyTorch CUDA version:"
python -c "import torch; print('CUDA version:', torch.version.cuda)"

cd /leonardo/home/userexternal/fmorandi/MOTR_domain_gap

export WANDB_MODE=offline

# Lancia il training sul dataset mot17
python3 -m torch.distributed.run --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr="localhost" --master_port=12345 \
    main.py  \
    --meta_arch motr \
    --dataset_file e2e_joint \
    --epoch 50 \
    --with_box_refine \
    --lr_drop 40 \
    --lr 2e-4 \
    --lr_backbone 2e-5 \
    --pretrained ./pretrained/r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint.pth\
    --output_dir exps/e2e_motr_r50_mot17trainhalf \
    --batch_size 1 \
    --sample_mode 'random_interval' \
    --sample_interval 10 \
    --sampler_steps 10 20 30 \
    --sampler_lengths 2 3 4 5 \
    --update_query_pos \
    --merger_dropout 0 \
    --dropout 0 \
    --random_drop 0.1 \
    --fp_ratio 0.3 \
    --query_interaction_layer 'QIM' \
    --extra_track_attn \
    --mot_path ./datasets/mot \
    --data_txt_path_train ./datasets/data_path/mot17_half_train.train \
    --data_txt_path_val ./datasets/data_path/mot17_half_val.train \
