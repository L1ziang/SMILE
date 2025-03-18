targets=$(echo {1..50} | tr ' ' ',')

# Mirror-w
python my_whitebox_attacks.py --attack_mode Mirror-w --target_dataset vggface2 --dataset celeba_partial256 --arch_name_target inception_resnetv1_vggface2 --test_target=$targets --epochs 20000 --population_size 100000 --test_only

# PPA
python my_whitebox_attacks.py --attack_mode PPA --target_dataset vggface2 --dataset celeba_partial256 --arch_name_target inception_resnetv1_vggface2 --test_target=$targets --population_size 5000 --epochs 70 --candidate 200 --final_selection 50 --iterations 100 --lr 0.005 --test_only

# Mirror-b
python my_blackbox_attacks.py --attack_mode Mirror-b --target_dataset vggface2 --dataset celeba_partial256 --arch_name_target inception_resnetv1_vggface2 --test_target=$targets --population_size 1000 --generations 20 --test_only

# RLBMI
python my_blackbox_attacks.py --attack_mode RLB-MI --target_dataset vggface2 --dataset celeba_RLBMI --arch_name_target inception_resnetv1_vggface2 --test_target=$targets --max_episodes 40000 --test_only

# SMILE
python my_blackbox_attacks.py --attack_mode ours-surrogate_model --target_dataset vggface2 --dataset celeba_partial256 --arch_name_target inception_resnetv1_vggface2 --test_target=$targets --budget 1000 --population_size 2500 --finetune_mode 'vggface2->CASIA' --arch_name_finetune inception_resnetv1_casia --EorOG SMILE --epochs 200 --lr 0.2 --test_only



