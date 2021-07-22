# time python src/reinforcementlearning/softActorCritic/soft_actor_critic.py --root_dir=rs_01_et_1_obs_bc__even_sparser_reward --reward_scaling=0.1 --entropy_target=-2 --behavioral_cloning_checkpoint_dir=behavioral_cloning_obstacles_sparse
# time python src/reinforcementlearning/softActorCritic/soft_actor_critic.py --root_dir=rs_01_et_1_obs_bc_sparse_reward --reward_scaling=0.1 --entropy_target=-2 --behavioral_cloning_checkpoint_dir=behavioral_cloning_obstacles
# time python src/reinforcementlearning/softActorCritic/soft_actor_critic.py --root_dir=rs_01_et_1_obs_1_attr_bc --reward_scaling=0.1 --entropy_target=-2 --behavioral_cloning_checkpoint_dir=behavioral_cloning_obstacles


time python src/reinforcementlearning/softActorCritic/soft_actor_critic.py --root_dir=rs_01_new_scenarios --reward_scaling=0.1 --difficulty=med

cp -r src/reinforcementlearning/softActorCritic/checkpoints/rs_01_new_scenarios/ src/reinforcementlearning/softActorCritic/checkpoints/rs_01_new_scenarios_med_backup/

time python src/reinforcementlearning/softActorCritic/soft_actor_critic.py --root_dir=rs_01_new_scenarios --reward_scaling=0.1 --difficulty=hard

time python src/reinforcementlearning/softActorCritic/soft_actor_critic.py --root_dir=rs_01_new_scenarios_yolo_hard --reward_scaling=0.1 --difficulty=hard
