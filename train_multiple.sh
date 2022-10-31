CUDA_VISIBLE_DEVICES=3 exec python scripts/archetypal/distributed.py --port 12355 --directory /mntc/tanners/hatespace/checkpoints --latent_dim_size 512 --num_workers 4 2>&1 > /mntc/tanners/hatespace/terminal_outputs/terminal1.txt &
CUDA_VISIBLE_DEVICES=4 exec python scripts/archetypal/distributed.py --port 12354 --directory /mntc/tanners/hatespace/checkpoints --latent_dim_size 32 --num_workers 4 2>&1 > /mntc/tanners/hatespace/terminal_outputs/terminal2.txt &
CUDA_VISIBLE_DEVICES=6 exec python scripts/archetypal/distributed.py --port 12353 --directory /mntc/tanners/hatespace/checkpoints --latent_dim_size 16 --num_workers 4 2>&1 > /mntc/tanners/hatespace/terminal_outputs/terminal3.txt &
CUDA_VISIBLE_DEVICES=7 exec python scripts/archetypal/distributed.py --port 12352 --directory /mntc/tanners/hatespace/checkpoints --latent_dim_size 8 --num_workers 4 2>&1 > /mntc/tanners/hatespace/terminal_outputs/terminal4.txt

