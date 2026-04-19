#!/bin/bash
echo "📥 Pulling Results from Andes..."

#  runs every 30 seconds to save bandwidth
while true; do
    rsync -avz --update \
        --exclude '*.ckpt' \
        andes:/dartfs/rc/lab/F/FranklandS/tom/codebase/results/causal_analysis \
        ./results/

    rsync -avz --update \
        --exclude '*.pt' \
        --exclude '*.ckpt' \
        andes:/dartfs/rc/lab/F/FranklandS/tom/results/ \
        ./other_results/
    
    rsync -avz --update \
        --exclude '*.pt' \
        --exclude '*.ckpt' \
        andes:/dartfs/rc/lab/F/FranklandS/tom/logs/ \
        ./slurm_logs/
    
    echo "   [$(date '+%H:%M:%S')] Synced. Waiting 30s..."
    sleep 15
done