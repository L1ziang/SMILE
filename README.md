# _From Head to Tail: Efficient Black-box Model Inversion Attack via Long-tailed Learning_ - CVPR 2025

## 📄 [Paper](https://arxiv.org/abs/2503.16266)

## 📝 Abstract
_Model Inversion Attacks (MIAs) aim to reconstruct private training data from models, leading to privacy leakage, particularly in facial recognition systems. Although many studies have enhanced the effectiveness of white-box MIAs, less attention has been paid to improving efficiency and utility under limited attacker capabilities. Existing black-box MIAs necessitate an impractical number of queries, incurring significant overhead. Therefore, we analyze the limitations of existing MIAs and introduce **S**urrogate **M**odel-based **I**nversion with **L**ong-tailed **E**nhancement (**SMILE**), a high-resolution oriented and query-efficient MIA for the black-box setting. We begin by analyzing the initialization of MIAs from a data distribution perspective and propose a long-tailed surrogate training method to obtain high-quality initial points. We then enhance the attack's effectiveness by employing the gradient-free black-box optimization algorithm selected by NGOpt. Our experiments show that **SMILE** outperforms existing state-of-the-art black-box MIAs while requiring only about 5% of the query overhead._

## 📦 Environment Installation
```bash
conda env create -f environment.yml
```

## 🔍 Datasets and Models Download

- **Datasets** : refer to ./train_classification_models/README.md
- **Models** : refer to [./checkpoints](https://drive.google.com/drive/folders/1Ka5s0e8UdXKNUOFdIDBxfJAQ2TfiJG_r?usp=drive_link) & [./classification_models](https://drive.google.com/drive/folders/14I9n1pPuHWJiBbdhDTsaoFajSyoXMmvA?usp=drive_link)
- **conf_mask.pt** : refer to [./conf_mask.pt](https://drive.google.com/file/d/19QQE0DZffsdBFQv0lOad4U9T3a9O8XHF/view?usp=drive_link)

- **Pre-trained target models** : The GANs and classification model we use follows [MIRROR](https://github.com/njuaplusplus/mirror):

## 😃 SMILE

1.Initial sampling 2.5K synthetic images
```bash
python my_sample_z_w_space.py
```

2.Query the black-box target model to obtain the output
```bash
python my_generate_blackbox_attack_dataset.py --arch_name inception_resnetv1_vggface2 vggface2 celeba_partial256
```

3.Merge all tensors
```bash
python my_merge_all_tensors.py blackbox_attack_data/vggface2/inception_resnetv1_vggface2/celeba_partial256/
```

4.Long-tailed surrogate training
```bash
python long-tailed_surrogate_training.py --target_dataset vggface2 --dataset celeba_partial256 --arch_name_target inception_resnetv1_vggface2 --arch_name_finetune inception_resnetv1_casia --finetune_mode 'vggface2->CASIA' --epoch 200 --batch_size 128 --query_num 2500
```

5.Local White-box attacks & Gradient-free Black-Box attacks
```bash
run_SMILE.sh
```

Baselines:
```bash
run.sh
```

## 🔨 Evaluation for Attacks
Data generation for evaluation : gen_eval_data.py
```bash
test.sh
```

## 📚 Evaluation for Models
Datasets for evaluate the accuracy of surrogate models & Self-Trained Classification Models

refer to ./train_classification_models/README.md


## 🔥 Acknowledgement

The codebase is based on [MIRROR](https://github.com/njuaplusplus/mirror).

The StyleGAN models are based on [genforce/genforce](https://github.com/genforce/genforce).

VGG16/VGG16BN/Resnet50 models are from [their official websites](https://www.robots.ox.ac.uk/~albanie/pytorch-models.html).

InceptionResnetV1 is from [timesler/facenet-pytorch](https://github.com/timesler/facenet-pytorch).

SphereFace is from [clcarwin/sphereface_pytorch](https://github.com/clcarwin/sphereface_pytorch).

Our baselines are implemented based on the following repositories. We extend our gratitude to the authors for open-sourcing their code.
 [MIRROR](https://github.com/njuaplusplus/mirror), [PPA](https://github.com/LukasStruppek/Plug-and-Play-Attacks), [RLBMI](https://github.com/HanGyojin/RLB-MI)

## 📜 Citation

```
@article{li2025head,
  title={From Head to Tail: Efficient Black-box Model Inversion Attack via Long-tailed Learning},
  author={Li, Ziang and Zhang, Hongguang and Wang, Juan and Chen, Meihui and Hu, Hongxin and Yi, Wenzhe and Xu, Xiaoyang and Yang, Mengda and Ma, Chenjun},
  journal={arXiv preprint arXiv:2503.16266},
  year={2025}
}
```
