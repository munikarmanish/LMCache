#!/bin/bash

PYTHONHASHSEED=0 \
    lmcache_controller \
        --host 0.0.0.0 \
        --port 9000 \
        --monitor-ports '{"pull": 8300, "reply": 8400}'
