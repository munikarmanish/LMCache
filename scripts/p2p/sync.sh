#!/bin/bash

rsync -ruvhp --delete --exclude '*.log' ./ manish@192.168.128.32:code/test_lmc/p2p/
