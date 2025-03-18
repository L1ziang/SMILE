for target in {1..50}

do
    python my_whitebox_attacks.py --attack_mode ours-w --target_dataset vggface2 --dataset celeba_partial256 --arch_name_target inception_resnetv1_vggface2 --target $target --epochs 200 --arch_name_finetune inception_resnetv1_casia --finetune_mode 'vggface2->CASIA' --num_experts 3 --EorOG SMILE --population_size 2500
    python my_blackbox_attacks.py --attack_mode ours-surrogate_model --target_dataset vggface2 --dataset celeba_partial256 --arch_name_target inception_resnetv1_vggface2 --target $target --budget 1000 --population_size 2500 --epochs 200 --finetune_mode 'vggface2->CASIA' --arch_name_finetune inception_resnetv1_casia --EorOG SMILE --lr 0.2 --x 1.7
done
