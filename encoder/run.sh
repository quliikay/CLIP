CUDA_VISIBLE_DEVICES=1 python -u trojan.py --lr 1e-8 --lam 1e-1 --train_bs 256 --use_wandb --train_ratio 1e-2
CUDA_VISIBLE_DEVICES=1 python -u trojan.py --lr 1e-8 --lam 5e-1 --train_bs 256 --use_wandb --train_ratio 1e-2
CUDA_VISIBLE_DEVICES=1 python -u trojan.py --lr 1e-8 --lam 5e-2 --train_bs 256 --use_wandb --train_ratio 1e-2
CUDA_VISIBLE_DEVICES=1 python -u trojan.py --lr 1e-8 --lam 1e-2 --train_bs 256 --use_wandb --train_ratio 1e-2
CUDA_VISIBLE_DEVICES=1 python -u trojan.py --lr 1e-8 --lam 5e-3 --train_bs 256 --use_wandb --train_ratio 1e-2
