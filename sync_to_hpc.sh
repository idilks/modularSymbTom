revised command:
  while true; do rsync -avz \
    --exclude '.git' \
    --exclude '.vscode' \
    --exclude '.conda' \
    --exclude '.conda_backup_20250713_101749' \
    --exclude '__pycache__' \
    --exclude '.pytest_cache' \
    --exclude '.ipynb_checkpoints' \
    --exclude 'results' \
    --exclude 'wandb' \
    --exclude 'paper' \
    --exclude '*.pt' \
    --exclude '*.pth' \
    . andes:/dartfs/rc/lab/F/FranklandS/tom/; sleep 2; done