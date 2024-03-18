source scratch/bin/activate
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/11.8.0

srun -p elec.gpu.q --gres=gpu:1 --time=8:00:00 --pty bash
# srun -p elec.gpu-es02.q --gres=gpu:1 --time=8:00:00 --pty bash


# source multi_no-torch/bin/activate
# echo "Load module Pytorch/1.12.1-foss-2022a-CUDA-11.7.0"
# module load PyTorch/1.12.1-foss-2022a-CUDA-11.7.0
# module load cuda11.1


# input_command=$1
# if [ "$input_command" = "2080" ]; then
#     echo "startup gpu GTX2080 session, duration" $session_duration "hours"
#     # jupyter notebook --no-browser --port=8808 --ip=elec-gpuD001
#     srun -p elec.gpu.q --gres=gpu:1 --time=$session_duration:00:00 --pty bash
#     return
# fi


# if [ "$input_command" = "A100" ]; then
#     echo "startup gpu A100 session, duration" $session_duration "hours"
#     # jupyter notebook --no-browser --port=8808 --ip=elec-gpuD001
#     srun -p elec.gpu-es02.q --gres=gpu:1 --time=$session_duration:00:00 --pty bash    
#     return
# fi
