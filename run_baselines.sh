for target in {1..50}

do
    # Mirror-w
    python my_whitebox_attacks.py --attack_mode Mirror-w --target_dataset vggface2 --dataset celeba_partial256 --arch_name_target inception_resnetv1_vggface2 --target $target --epochs 20000 --population_size 100000

    # PPA
    python my_whitebox_attacks.py --attack_mode PPA --target_dataset vggface2 --dataset celeba_partial256 --arch_name_target inception_resnetv1_vggface2 --target $target --population_size 5000 --epochs 70 --candidate 200 --final_selection 50 --iterations 100 --lr 0.005

    # Mirror-b
    python my_blackbox_attacks.py --attack_mode Mirror-b --target_dataset vggface2 --dataset celeba_partial256 --arch_name_target inception_resnetv1_vggface2 --target $target --population_size 1000 --generations 20

    # RLBMI
    python my_blackbox_attacks.py --attack_mode RLB-MI --target_dataset vggface2 --dataset celeba_RLBMI --arch_name_target inception_resnetv1_vggface2 --target $target --max_episodes 40000

done
