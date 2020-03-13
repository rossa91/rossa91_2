# rossa91

| GPU | NETWORK | PERFORMACNE | 
|:-------|-------:|:------:|
|   Tesla P4    |   VGG9    |   15.4s   |
|   Tesla P4    |   QVGG9    |   25.3s   |

!python /content/main.py --num_bits 3 --smooth_grad True --qtype True --epoch 3 --mixed True --mask_load /content/checkpoint/tracking/track_section0_3/mask_0.1.pth
