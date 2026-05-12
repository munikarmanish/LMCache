#!/bin/bash

CONTROLLER="192.168.128.31:9000"

http "${CONTROLLER}/controller/key-stats"
