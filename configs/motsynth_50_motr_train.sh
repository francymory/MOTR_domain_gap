#!/bin/bash  
#SBATCH --job-name=MOTR_train_motsynth50
#SBATCH --output="/leonardo/home/userexternal/fmorandi/MOTR_domain_gap/exps/slurm_logs/%x_%A_%a.out"
#SBATCH --error="/leonardo/home/userexternal/fmorandi/MOTR_domain_gap/exps/slurm_logs/%x_%A_%a.err"
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=160GB
#SBATCH --cpus-per-gpu=8
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrB_FeeCO
#SBATCH --time=2-00:00:00
#SBATCH --qos=boost_qos_lprod
##SBATCH --qos=boost_qos_dbg

# Attivare l'ambiente virtuale
source /leonardo/home/userexternal/fmorandi/envs/motr/bin/activate
pwd

# Verifica che Conda sia attivato correttamente
which python  # Dovrebbe stampare il percorso dell'interprete Python
python --version  # Controlla che la versione di Python sia quella prevista

# Verifica che le GPU siano disponibili
echo ">>> GPU disponibili:"
nvidia-smi  # Mostra le GPU disponibili

# Verifica che PyTorch rilevi le GPU
echo ">>> PyTorch CUDA availability:"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Verifica la versione di CUDA utilizzata da PyTorch
echo ">>> PyTorch CUDA version:"
python -c "import torch; print('CUDA version:', torch.version.cuda)"

cd /leonardo/home/userexternal/fmorandi/MOTR_domain_gap

# Configura le variabili di ambiente per il training distribuito
export WANDB_MODE=offline
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=3600

# Impostazione dinamica di MASTER_ADDR e MASTER_PORT per la comunicazione tra nodi
IFS=',' read -r -a nodelist <<<$SLURM_NODELIST
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(comm -23 <(seq 5000 6000 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

# Mostra alcune informazioni di configurazione
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"

# Variabili per il modello pre-addestrato e la directory di output
pretrain="/leonardo/home/userexternal/fmorandi/MOTR_domain_gap/pretrained/r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint.pth"
output_dir="/leonardo/home/userexternal/fmorandi/MOTR_domain_gap/exps/e2e_motr_r50_motsynth_train50"

# Lancia il training usando torchrun per il training distribuito
srun --exclusive -c $SLURM_CPUS_PER_GPU torchrun --nnodes=$SLURM_NNODES --nproc_per_node=$SLURM_GPUS_PER_NODE --rdzv-endpoint=$MASTER_ADDR --rdzv-id=$SLURM_JOB_NAME --rdzv-backend=c10d main.py \
    --meta_arch motr \
    --dataset_file e2e_motsynth \
    --epoch 20 \
    --with_box_refine \
    --lr_drop 16 \
    --lr 2e-4 \
    --lr_backbone 2e-5 \
    --pretrained $pretrain\
    --output_dir $output_dir\
    --batch_size 1 \
    --sample_mode 'random_interval' \
    --sample_interval 5 \
    --sampler_steps 5 10 12 \
    --sampler_lengths 2 3 4 5 \
    --update_query_pos \
    --merger_dropout 0 \
    --dropout 0 \
    --random_drop 0.1 \
    --fp_ratio 0.3 \
    --query_interaction_layer 'QIM' \
    --extra_track_attn \
    --mot_path ./datasets/mot \
    --data_txt_path_train ./datasets/data_path/motsynth_50_train_correct.train\
    --data_txt_path_val ./datasets/data_path/motsynth_50_val.train
