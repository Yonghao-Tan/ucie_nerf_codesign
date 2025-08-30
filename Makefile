# test:
# 	cd ./eval && \
# 	CUDA_VISIBLE_DEVICES=6 python eval.py --config ../configs/eval_llff.txt
# 	# --weight_sample_fine --coarse_sampling_mask --sample_block_size 5 --moe --sv_prune --sparse

test:
	cd ./eval && \
	CUDA_VISIBLE_DEVICES=6 python eval.py --config ../configs/eval_llff.txt --num_source_views 8 --chunk_height 15 --eval_scenes fern

test2:
	cd ./eval && \
	CUDA_VISIBLE_DEVICES=6 python eval.py --config ../configs/eval_llff_sr.txt --eval_scenes fern --chunk_height 10 --num_source_views 8 --ckpt_path ../pretrained/model_share_svprune_5_sr_w8a8_s50.pth --resize_factor 0.5 --sr --window_size 5 --sample_point_sparsity --sv_prune --sv_top_k 5 --sparsity 0.5 --sample_point_group_size 8 --q_bits 8 --use_moe

tests:
	cd ./eval && \
	CUDA_VISIBLE_DEVICES=6 python eval.py --config ../configs/eval_llff_sr.txt --ckpt_path ../pretrained/model_16_32_sr_tuned.pth --chunk_height 10 --window_size 5 --num_source_views 8 --resize_factor 0.5 --sr --eval_scenes fern --q_bits 8 

train:
	TORCH_DISTRIBUTED_DEBUG=DETAIL CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -m torch.distributed.launch --nproc_per_node=6 train.py --config configs/pretrain.txt --no_load_scheduler --no_load_opt --ckpt_path pretrained/model_16_32.pth --sample_mode block_single --use_moe --sample_point_sparsity

train2:
	TORCH_DISTRIBUTED_DEBUG=DETAIL CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -m torch.distributed.launch --master_port=33111 --nproc_per_node=6 train.py --config configs/pretrain_sr.txt --ckpt_path pretrained/model_16_32_sr_tuned.pth --no_load_scheduler --no_load_opt --chunk_height 10 --sample_mode block_single --resize_factor 0.5 --sr --num_source_views 8 --window_size 5 --sample_point_sparsity --sv_prune --sv_top_k 5 --sparsity 0.6 --use_moe --sample_point_group_size 8 --q_bits 8
# --sample_point_sparsity  --sv_prune --sv_top_k 5 --sr_sparsity 0.25 --sparsity 0.6 --use_moe