#!/usr/bin/env bash
cd eval/

# CUDA_VISIBLE_DEVICES=0 python eval.py --config ../configs/eval_nerf.txt --num_source_views 8 --eval_scenes mic &
# CUDA_VISIBLE_DEVICES=1 python eval.py --config ../configs/eval_nerf.txt --num_source_views 8 --eval_scenes chair &
# CUDA_VISIBLE_DEVICES=2 python eval.py --config ../configs/eval_nerf.txt --num_source_views 8 --eval_scenes lego &
# CUDA_VISIBLE_DEVICES=3 python eval.py --config ../configs/eval_nerf.txt --num_source_views 8 --eval_scenes ficus &
# CUDA_VISIBLE_DEVICES=4 python eval.py --config ../configs/eval_nerf.txt --num_source_views 8 --eval_scenes materials &
# CUDA_VISIBLE_DEVICES=5 python eval.py --config ../configs/eval_nerf.txt --num_source_views 8 --eval_scenes hotdog &
# CUDA_VISIBLE_DEVICES=6 python eval.py --config ../configs/eval_nerf.txt --num_source_views 8 --eval_scenes ship &
# CUDA_VISIBLE_DEVICES=6 python eval.py --config ../configs/eval_nerf.txt --num_source_views 8 --eval_scenes drums &

CUDA_VISIBLE_DEVICES=0 python eval.py --config ../configs/eval_nerf_sr.txt --eval_scenes mic --ckpt_path ../pretrained/model_16_32_sr_small.pth --chunk_height 10 --window_size 5 --num_source_views 8  &
CUDA_VISIBLE_DEVICES=1 python eval.py --config ../configs/eval_nerf_sr.txt --eval_scenes chair --ckpt_path ../pretrained/model_16_32_sr_small.pth --chunk_height 10 --window_size 5 --num_source_views 8  &
CUDA_VISIBLE_DEVICES=2 python eval.py --config ../configs/eval_nerf_sr.txt --eval_scenes lego --ckpt_path ../pretrained/model_16_32_sr_small.pth --chunk_height 10 --window_size 5 --num_source_views 8  &
CUDA_VISIBLE_DEVICES=3 python eval.py --config ../configs/eval_nerf_sr.txt --eval_scenes ficus --ckpt_path ../pretrained/model_16_32_sr_small.pth --chunk_height 10 --window_size 5 --num_source_views 8  &
CUDA_VISIBLE_DEVICES=4 python eval.py --config ../configs/eval_nerf_sr.txt --eval_scenes materials --ckpt_path ../pretrained/model_16_32_sr_small.pth --chunk_height 10 --window_size 5 --num_source_views 8  &
CUDA_VISIBLE_DEVICES=5 python eval.py --config ../configs/eval_nerf_sr.txt --eval_scenes hotdog --ckpt_path ../pretrained/model_16_32_sr_small.pth --chunk_height 10 --window_size 5 --num_source_views 8  &
CUDA_VISIBLE_DEVICES=6 python eval.py --config ../configs/eval_nerf_sr.txt --eval_scenes ship --ckpt_path ../pretrained/model_16_32_sr_small.pth --chunk_height 10 --window_size 5 --num_source_views 8  &
CUDA_VISIBLE_DEVICES=6 python eval.py --config ../configs/eval_nerf_sr.txt --eval_scenes drums --ckpt_path ../pretrained/model_16_32_sr_small.pth --chunk_height 10 --window_size 5 --num_source_views 8  &