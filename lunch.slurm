#!/bin/bash
#SBATCH --job-name=compute-information                 # create a short name for your job
#SBATCH --output="compute-information-%j.out"
#SBATCH --partition=gpu
#SBATCH --qos=gpu085862023
#SBATCH --gres=gpu


source venv/bin/activate
module load cuda

#for at in 10 100 1000 10000
#do
#       python benchmark_sparse_retrieval_from_pyterrier_bow.py beir_datasets/msmarco $at
#       python benchmark_sparse_retrieval_from_pyterrier_bow.py beir_datasets/msmarco $at --algorithm iterative --objective half --fp_16
        #python benchmark_sparse_retrieval_from_splade.py beir_datasets/msmarco 1000
#python benchmark_sparse_retrieval_from_splade.py beir_datasets/msmarco 1000 --algorithm iterative --objective half
#done

python compute_information.py clean_datasets/politics__speeches_clean.jsonl EleutherAI/gpt-neo-125m --context-percentage 0.25 --dtype float32
python compute_information.py clean_datasets/politics__speeches_clean.jsonl EleutherAI/gpt-neo-125m --context-percentage 0.25 --dtype float16
python compute_information.py clean_datasets/politics__speeches_clean.jsonl EleutherAI/gpt-neo-125m --context-percentage 0.25 --dtype bfloat16
python compute_information.py clean_datasets/politics__speeches_clean.jsonl EleutherAI/gpt-neo-125m --context-percentage 0.25 --dtype int8
python compute_information.py clean_datasets/politics__speeches_clean.jsonl EleutherAI/gpt-neo-125m --context-percentage 0.25 --dtype int4


deactivate    