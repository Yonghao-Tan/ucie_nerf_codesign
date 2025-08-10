#!/usr/bin/env bash
cd eval/

# 使用同一个GPU（假设是GPU 0）
CUDA_VISIBLE_DEVICES=6 python eval.py --config ../configs/eval_llff_sr.txt --resize_factor 0.5 --sr --eval_scenes fern &&
CUDA_VISIBLE_DEVICES=6 python eval.py --config ../configs/eval_llff_sr.txt --resize_factor 0.5 --sr --eval_scenes trex &&
CUDA_VISIBLE_DEVICES=6 python eval.py --config ../configs/eval_llff_sr.txt --resize_factor 0.5 --sr --eval_scenes room &&
CUDA_VISIBLE_DEVICES=6 python eval.py --config ../configs/eval_llff_sr.txt --resize_factor 0.5 --sr --eval_scenes flower &&
CUDA_VISIBLE_DEVICES=6 python eval.py --config ../configs/eval_llff_sr.txt --resize_factor 0.5 --sr --eval_scenes orchids &&
CUDA_VISIBLE_DEVICES=6 python eval.py --config ../configs/eval_llff_sr.txt --resize_factor 0.5 --sr --eval_scenes leaves &&
CUDA_VISIBLE_DEVICES=6 python eval.py --config ../configs/eval_llff_sr.txt --resize_factor 0.5 --sr --eval_scenes horns &&
CUDA_VISIBLE_DEVICES=6 python eval.py --config ../configs/eval_llff_sr.txt --resize_factor 0.5 --sr --eval_scenes fortress
