python train.py \
        --cuda \
        -d coco \
        -ms \
        -bs 16 \
        -accu 4 \
        --lr 0.001 \
        --max_epoch 250 \
        --lr_epoch 150 200 \
        