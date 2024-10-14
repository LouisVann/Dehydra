# Neural Dehydration

Source code for "Neural Dehydration: Effective Erasure of Black-box Watermarks from DNNs with Limited Data" (CCS '24 accepted).

## Datasets and Models

The experiments are performed under MNIST, CIFAR-10 and CIFAR-100 datasets, which target models LeNet-5, ResNet-18 and ResNet-34.

For instance, to train a clean ResNet-18 model without watermarks on CIFAR-10, please run

```
python train_regular.py --dataset cifar10 --arch resnet18
```

## Target Watermark Schemes

Our evaluation covers ten mainstream black-box DNN watermarks under `watermarks` directory, including five fixed-class watermarks and five non-fixed-class watermarks.

* **Fixed-class watermarks**: `protecting_ip.py` (*Content*, *Noise*, *Unrelated*), `piracy_resistant.py` (*Piracy*), `ewe.py` (*EWE*).
* **Non-fixed-class watermarks**: `weakness_into_strength.py` (*Adi*), `frontier_stitching.py` (*AFS*), `exponential_weighting.py` (*EW*), `blind.py` (*Blind*), `wm_embedded_systems.py` (*Mark*).

For instance, to embed a *Content* watermark into ResNet-18 model on CIFAR-10, please run

```
python gen_watermarks.py --arch resnet18 --method ProtectingIP --wm_type content --dataset cifar10 --trg_set_size 100 --save_wm
```

, and then

```
python train_watermark.py --lr 0.01 --method ProtectingIP --wm_type content --dataset cifar10 --arch resnet18 --epochs_w_wm 40 --sched CosineAnnealingLR
```

## Attacks

Our Dehydra attack is implemented under `attacks` directory.

For instance, to launch the basic Dehydra against the above target model under the in-distribution data setting, run

```
python attack.py --attack_type inversion --inverse_num 250 --new_weight 15 --lr 0.01 --data_ratio 0.01667 --dataset cifar10 --arch resnet18 --method ProtectingIP --wm_type content --batch_size 128
```

To launch the improved Dehydra, please first run the above command to generate class-wise recovered samples. Then, detect the potential fixed-class using

```
python target_label_detection_inverse.py --dataset cifar10 --method ProtectingIP --wm_type content
```

, and finally perform the removal attack

```
python attack.py --attack_type improved --re_batch_size 128 --lr 0.01 --data_ratio 0.01667 --dataset cifar10 --arch resnet18 --method ProtectingIP --wm_type content --fixed
```