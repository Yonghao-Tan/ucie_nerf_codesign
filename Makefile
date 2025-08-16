# test:
# 	cd ./eval && \
# 	CUDA_VISIBLE_DEVICES=6 python eval.py --config ../configs/eval_llff.txt
# 	# --weight_sample_fine --coarse_sampling_mask --sample_block_size 5 --moe --sv_prune --sparse

test:
	cd ./eval && \
	CUDA_VISIBLE_DEVICES=0 python eval.py --config ../configs/eval_llff.txt --num_source_views 8

test2:
	cd ./eval && \
	CUDA_VISIBLE_DEVICES=6 python eval.py --config ../configs/eval_llff_sr.txt --num_source_views 8 --resize_factor 0.5 --sr --sample_point_sparsity --use_moe --sv_prune --sv_top_k 5 --sample_point_group_size 4
# --sample_point_sparsity --use_moe --sv_prune --sv_top_k 5 --sample_point_group_size 4

tests:
	cd ./eval && \
	CUDA_VISIBLE_DEVICES=1 python eval.py --config ../configs/eval_llff_sr.txt --num_source_views 8 --resize_factor 0.5 --sr

train:
	TORCH_DISTRIBUTED_DEBUG=DETAIL CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -m torch.distributed.launch --nproc_per_node=6 train.py --config configs/pretrain.txt --no_load_scheduler --no_load_opt --ckpt_path pretrained/model_16_32.pth --sample_mode block_single --use_moe --sample_point_sparsity

traint:
	TORCH_DISTRIBUTED_DEBUG=DETAIL CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -m torch.distributed.launch --master_port=33111 --nproc_per_node=2 train.py --config configs/pretrain_sr.txt --ckpt_path pretrained/model_16_32_sr_tuned.pth --no_load_scheduler --no_load_opt --sample_mode block_single --resize_factor 0.5 --sr

train2:
	TORCH_DISTRIBUTED_DEBUG=DETAIL CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -m torch.distributed.launch --master_port=33111 --nproc_per_node=6 train.py --config configs/pretrain_sr.txt --ckpt_path pretrained/model_16_32_sr_tuned.pth --no_load_scheduler --no_load_opt --sample_mode block_single --resize_factor 0.5 --sr --num_source_views 8 --sample_point_sparsity --use_moe --sv_prune --sv_top_k 5 --sample_point_group_size 4
# --use_moe --sample_point_sparsity 