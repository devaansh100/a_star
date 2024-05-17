#!/bin/bash
echo "GPU 0"
CUDA_VISIBLE_DEVICES=0 python -c "import torch;torch.tensor(1).cuda()"
echo "GPU 1"
CUDA_VISIBLE_DEVICES=1 python -c "import torch;torch.tensor(1).cuda()"
echo "GPU 2"
CUDA_VISIBLE_DEVICES=2 python -c "import torch;torch.tensor(1).cuda()"
echo "GPU 3"
CUDA_VISIBLE_DEVICES=3 python -c "import torch;torch.tensor(1).cuda()"
