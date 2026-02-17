ToM:
	python3  backend/rl-multistew/server.py \
		--max-steps 1000000 \
		--num_agents 1 \
		--num_envs 1 \
		--num_steps 128 \
		--seed 1 \
		--layout overcooked_cramped_room_v0 \
		--num_mini_batches 4 \
		--ppo_epochs 10 \
		--clip_param 0.2 \
		--value_loss_coef 0.5 \
		--entropy_coef 0.01 \
		--max_grad_norm 0.5 \
		--gamma 0.99 \
		--lam 0.95 \
		--data_path backend/rl-multistew/data_local \
		--feature tom_full \
		--log \
		#--model_path backend/rl-multistew/models_0829/2_agents_overcooked_cramped_room_v0_seed_2_global_RS_100.pth \
		#--checkpoint_dir models_fake \
		--region_size 100
cramped:
	python3  backend/rl-multistew/server.py \
		--max-steps 10000 \
		--num_agents 2 \
		--num_envs 1 \
		--num_steps 256 \
		--seed 1 \
		--layout overcooked_cramped_room_v0 \
		--num_mini_batches 4 \
		--ppo_epochs 5 \
		--clip_param 0.2 \
		--value_loss_coef 0.5 \
		--entropy_coef 0.01 \
		--max_grad_norm 0.5 \
		--gamma 0.99 \
		--lam 0.95 \
		--data_path backend/rl-multistew/data_local \
		--feature global \
		--model_path backend/rl-multistew/models_0829/2_agents_overcooked_cramped_room_v0_seed_2_global_RS_100.pth \
		--checkpoint_dir models_fake \
		--region_size 100
cramped3:
	python3  backend/rl-multistew/server.py \
		--max-steps 10000 \
		--num_agents 2 \
		--num_envs 1 \
		--num_steps 256 \
		--seed 1 \
		--layout overcooked_cramped_room_v0 \
		--num_mini_batches 4 \
		--ppo_epochs 5 \
		--clip_param 0.2 \
		--value_loss_coef 0.5 \
		--entropy_coef 0.01 \
		--max_grad_norm 0.5 \
		--gamma 0.99 \
		--lam 0.95 \
		--data_path backend/rl-multistew/data_local \
		--feature local \
		--model_path backend/rl-multistew/models_0913/2_agents_overcooked_cramped_room_v3_seed_2_local_RS_100.pth \
		--checkpoint_dir models_fake \
		--region_size 100
forced:
	python3  backend/rl-multistew/server.py \
		--max-steps 10000 \
		--num_agents 2 \
		--num_envs 1 \
		--num_steps 256 \
		--seed 1 \
		--layout overcooked_cramped_room_v0 \
		--num_mini_batches 4 \
		--ppo_epochs 5 \
		--clip_param 0.2 \
		--value_loss_coef 0.5 \
		--entropy_coef 0.01 \
		--max_grad_norm 0.5 \
		--gamma 0.99 \
		--lam 0.95 \
		--data_path backend/rl-multistew/data_local \
		--feature global \
		--model_path backend/rl-multistew/models_0829/2_agents_overcooked_forced_coordination_v0_seed_2_global_RS_100.pth \
		--checkpoint_dir models_fake \
		--region_size 100
ring:
	python3  backend/rl-multistew/server.py \
		--max-steps 5000 \
		--num_agents 2\
		--num_envs 1 \
		--num_steps 256 \
		--seed 1 \
		--layout ring \
		--num_mini_batches 4 \
		--ppo_epochs 5 \
		--clip_param 0.2 \
		--value_loss_coef 0.5 \
		--entropy_coef 0.01 \
		--max_grad_norm 0.5 \
		--gamma 0.99 \
		--lam 0.95   \
		--data_path backend/rl-multistew/data_local \
		--feature local \
		--model_path backend/rl-multistew/models_0829/2_agents_overcooked_coordination_ring_v0_seed_2_local_RS_2.pth \
		--region_size 2
counter:
	python3  backend/rl-multistew/server.py \
		--max-steps 10000 \
		--num_agents 2\
		--num_envs 1 \
		--num_steps 256 \
		--seed 1 \
		--layout ring \
		--num_mini_batches 4 \
		--ppo_epochs 5 \
		--clip_param 0.2 \
		--value_loss_coef 0.5 \
		--entropy_coef 0.01 \
		--max_grad_norm 0.5 \
		--gamma 0.99 \
		--lam 0.95   \
		--data_path backend/rl-multistew/data_local \
		--feature local \
		--model_path backend/rl-multistew/models_0829/2_agents_overcooked_counter_circuit_v0_seed_2_local_RS_3.pth \
		--region_size 3

asym:
	python3  backend/rl-multistew/server.py \
		--max-steps 10000 \
		--num_agents 2\
		--num_envs 1 \
		--num_steps 256 \
		--seed 1 \
		--layout ring \
		--num_mini_batches 4 \
		--ppo_epochs 5 \
		--clip_param 0.2 \
		--value_loss_coef 0.5 \
		--entropy_coef 0.1 \
		--max_grad_norm 0.5 \
		--gamma 0.99 \
		--lam 0.95   \
		--data_path backend/rl-multistew/data_local \
		--feature local \
		--model_path backend/rl-multistew/models_0829/2_agents_overcooked_asymmetric_advantages_v0_seed_2_local_RS_4.pth \
		--region_size 4

asym_highlights:
	python3  backend/rl-multistew/server.py \
		--max-steps 20 \
		--num_agents 2\
		--num_envs 1 \
		--num_steps 256 \
		--seed 1 \
		--layout ring \
		--num_mini_batches 4 \
		--ppo_epochs 5 \
		--clip_param 0.2 \
		--value_loss_coef 0.5 \
		--entropy_coef 0.1 \
		--max_grad_norm 0.5 \
		--gamma 0.99 \
		--lam 0.95   \
		--data_path backend/rl-multistew/asym_hightlights \
		--feature local \
		--model_path backend/rl-multistew/models_0829/2_agents_overcooked_asymmetric_advantages_v0_seed_2_local_RS_100.pth \
		--region_size 100 \
		--log \
		--hightlights \
		--hightlights_k 5 \
		--hightlights_l 50 \
		--hightlights_interval_size 50 \
		--hightlights_states_after 10 
		
asym_image:
	python3  backend/rl-multistew/server.py \
		--max-steps 500 \
		--num_agents 2\
		--num_envs 1 \
		--num_steps 256 \
		--seed 1 \
		--layout asym \
		--num_mini_batches 4 \
		--ppo_epochs 5 \
		--clip_param 0.2 \
		--value_loss_coef 0.5 \
		--entropy_coef 0.1 \
		--max_grad_norm 0.5 \
		--gamma 0.99 \
		--lam 0.95   \
		--data_path backend/rl-multistew/asym_hightlights_image \
		--feature image \
		--model_path backend/rl-multistew/CNN_models/2_agents_overcooked_asymmetric_advantages_v0_seed_1_image_RS_2_agents_vision_"[100, 100]"_modelT_CNNAgent_hiddenD_256_numH_1_FS_1_TS_16.pth \
		--region_size 2 \
		--log \
		--saliency_endpoint \
		--hightlights_k 5 \
		--hightlights_l 30 \
		--hightlights_interval_size 20 \
		--hightlights_states_after 5 \
		--grid_width 11 \
		--grid_height 7 \
		--frame_stack 1 \
		--tile_size 16 \
		--lr 1e-4 \
		--model_type CNNAgent
ring_image:
	python3  backend/rl-multistew/server.py \
		--max-steps 200 \
		--num_agents 2\
		--num_envs 1 \
		--num_steps 256 \
		--seed 1 \
		--layout ring \
		--num_mini_batches 4 \
		--ppo_epochs 5 \
		--clip_param 0.2 \
		--value_loss_coef 0.5 \
		--entropy_coef 0.1 \
		--max_grad_norm 0.5 \
		--gamma 0.99 \
		--lam 0.95   \
		--data_path backend/rl-multistew/ring_hightlights_image \
		--feature image \
		--model_path backend/rl-multistew/CNN_models/2_agents_overcooked_coordination_ring_v0_seed_1_image_RS_2_agents_vision_"[100, 100]"_modelT_CNNAgent_hiddenD_256_numH_1_FS_1_TS_16.pth \
		--region_size 2 \
		--log \
		--normalize_advantages \
		--clip_vloss \
		--hightlights_k 5 \
		--hightlights_l 50 \
		--hightlights_interval_size 50 \
		--hightlights_states_after 10 \
		--grid_width 7 \
		--grid_height 7 \
		--frame_stack 1 \
		--tile_size 16 \
		--lr 1e-4 \
		--model_type CNNAgent

counter_image:
	python3  backend/rl-multistew/server.py \
		--max-steps 500 \
		--num_agents 2\
		--num_envs 1 \
		--num_steps 256 \
		--seed 1 \
		--layout counter_circuit \
		--num_mini_batches 4 \
		--ppo_epochs 5 \
		--clip_param 0.2 \
		--value_loss_coef 0.5 \
		--entropy_coef 0.1 \
		--normalize_advantages \
		--max_grad_norm 0.5 \
		--gamma 0.99 \
		--lam 0.95   \
		--data_path backend/rl-multistew/counter_hightlights_image \
		--feature image \
		--model_path backend/rl-multistew/CNN_models/2_agents_overcooked_counter_circuit_v0_seed_2_image_RS_4_agents_vision_"[100, 100]"_modelT_CNNAgent_hiddenD_256_numH_1_FS_1_TS_16.pth \
		--region_size 4 \
		--log \
		--saliency_endpoint \
		--hightlights_k 5 \
		--hightlights_l 50 \
		--hightlights_interval_size 50 \
		--hightlights_states_after 10 \
		--grid_width 10 \
		--grid_height 7 \
		--checkpoint_dir backend/rl-multistew/models_fake \
		--frame_stack 1 \
		--tile_size 16 \
		--lr 1e-4 \
		--model_type CNNAgent

forced_image:
	python3  backend/rl-multistew/server.py \
		--max-steps 500 \
		--num_agents 2\
		--num_envs 1 \
		--num_steps 256 \
		--seed 1 \
		--layout forced \
		--num_mini_batches 4 \
		--ppo_epochs 5 \
		--clip_param 0.2 \
		--value_loss_coef 0.5 \
		--entropy_coef 0.1 \
		--normalize_advantages \
		--max_grad_norm 0.5 \
		--gamma 0.99 \
		--lam 0.95   \
		--data_path backend/rl-multistew/forced_hightlights_agent0 \
		--feature image \
		--model_path backend/rl-multistew/CNN_models/2_agents_overcooked_forced_coordination_v0_seed_1_image_RS_2_agents_vision_"[100, 100]"_modelT_CNNAgent_hiddenD_256_numH_1_FS_1_TS_16.pth \
		--region_size 2 \
		--log \
		--hightlights_k 5 \
		--hightlights_l 50 \
		--hightlights_interval_size 50 \
		--hightlights_states_after 10 \
		--grid_width 7 \
		--grid_height 7 \
		--checkpoint_dir backend/rl-multistew/models_fake \
		--saliency_endpoint \	
		--frame_stack 1 \
		--tile_size 16 \
		--lr 1e-4 \
		--model_type CNNAgent
	
asym_image_layernorm:
	python3  backend/rl-multistew/server.py \
		--max-steps 1000 \
		--num_agents 2\
		--num_envs 1 \
		--num_steps 256 \
		--seed 1 \
		--layout ring \
		--num_mini_batches 4 \
		--ppo_epochs 5 \
		--clip_param 0.2 \
		--value_loss_coef 0.5 \
		--entropy_coef 0.1 \
		--max_grad_norm 0.5 \
		--gamma 0.99 \
		--lam 0.95   \
		--data_path backend/rl-multistew/asym_hightlights_image \
		--feature image \
		--model_path backend/rl-multistew/CNN_models/2_agents_overcooked_asymmetric_advantages_v0_seed_1_image_RS_100_agents_vision_"[100, 100]"_modelT_CNNAgent_hiddenD_256_numH_1_FS_1_TS_16.pth \
		--region_size 100 \
		--log \
		--highlights \
		--hightlights_k 5 \
		--hightlights_l 30 \
		--hightlights_interval_size 20 \
		--hightlights_states_after 5 \
		--grid_width 11 \
		--grid_height 7 \
		--frame_stack 1 \
		--tile_size 16 \
		--lr 1e-4 \
		--model_type LayerNormCNNAgent
ring_image:
	python3  backend/rl-multistew/server.py \
		--max-steps 200 \
		--num_agents 2\
		--num_envs 1 \
		--num_steps 256 \
		--seed 1 \
		--layout ring \
		--num_mini_batches 4 \
		--ppo_epochs 5 \
		--clip_param 0.2 \
		--value_loss_coef 0.5 \
		--entropy_coef 0.1 \
		--max_grad_norm 0.5 \
		--gamma 0.99 \
		--lam 0.95   \
		--data_path backend/rl-multistew/ring_hightlights_image \
		--feature image \
		--model_path backend/rl-multistew/CNN_models/2_agents_overcooked_coordination_ring_v0_seed_1_image_RS_100_agents_vision_"[100, 100]"_modelT_CNNAgent_hiddenD_256_numH_1_FS_1_TS_16.pth \
		--region_size 100 \
		--log \
		--normalize_advantages \
		--clip_vloss \
		--saliency_endpoint \	
		--hightlights_k 5 \
		--hightlights_l 50 \
		--hightlights_interval_size 50 \
		--hightlights_states_after 10 \
		--grid_width 7 \
		--grid_height 7 \
		--frame_stack 1 \
		--tile_size 16 \
		--lr 1e-4 \
		--model_type CNNAgent

gen_gif:
	python3 backend/rl-multistew/rl_multistew/utils.py --file_path backend/rl-multistew/asym_hightlights_agent0/2_agents_ring_seed_1_local_region_size_100_topk_trajectories.json 
gen_gif_npy:
	python3 backend/rl-multistew/rl_multistew/utils.py --npy_folder backend/rl-multistew/asym_hightlights_image

#	--num_steps "$4" \
#	--seed "$5" \
#	--layout "$6" \
#	--num_mini_batches "$7" \
#	--ppo_epochs "$8" \
#	--clip_param "$9" \
#	--value_loss_coef "${10}" \
#	--entropy_coef "${11}" \
#	--max_grad_norm "${12}" \
#	--gamma "${13}" \
#	--lam "${14}" \
#    --log \
#	--data_path "${15}" \
#	--feature "${16}"&
