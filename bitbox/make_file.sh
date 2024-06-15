#!/bin/bash

[ $# -ne 2 ] && echo Usage $0 FILENAME PAGE_COUNT && exit 1

dd if=/dev/zero of=$1 bs=4096 count=$blocks
