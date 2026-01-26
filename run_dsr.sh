#!/bin/bash
# DSR config matching successful Q-learning on meister_trimmed

python sac/run.py -m dsr \
    -lr 0.0003 \
    -bs 64 \
    -gamma 0.99 \
    -eps 1.0 \
    -eps_decay 0.995 \
    -tuf 50 \
    -rbs 50000 \
    -burnin 5000 \
    -num_ep 10000 \
    -timeout 500 \
    -fd 128 \
    -opt 10.0 \
    -mgn 10.0 \
    -conv \
    -rep pixel \
    -map meister_trimmed.txt \
    -map_yaml meister_trimmed.yaml \
    -test_map_yaml test_meister_trimmed.yaml \
    $@
