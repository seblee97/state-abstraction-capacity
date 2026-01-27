#!/bin/bash
# DSR config matching successful Q-learning on meister_trimmed

python sac/run.py -m dsr \
    -lr 0.0003 \
    -bs 64 \
    -gamma 0.99 \
    -eps 1.0 \
    -eps_decay 0.99998 \
    -recon \
    -rc 0.1 \
    -tuf 1000 \
    -rbs 50000 \
    -burnin 5000 \
    -num_ep 1000 \
    -timeout 50 \
    -fd 128 \
    -opt 0.0 \
    -mgn 1.0 \
    -conv \
    -rep pixel \
    -map map.txt \
    -map_yaml test_map.yaml \
    -test_map_yaml test_map.yaml \
    $@
