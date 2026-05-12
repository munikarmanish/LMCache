#!/bin/bash

CONTROLLER="192.168.128.31:9000"

clear_cache() {
    local instance="$1"

    http POST "${CONTROLLER}/clear" \
        instance_id="${instance}" \
        location=LocalCPUBackend
}

clear_cache "instance1" &>/dev/null
clear_cache "instance2" &>/dev/null

sleep 2

http "${CONTROLLER}/controller/key-stats"
