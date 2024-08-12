#!/bin/bash

folder=`dirname $0`
cd ${folder}/../..
echo "Current directory: $(pwd)"

mv /WatChMaL/folders_to_move/index_lists .
mv /WatChMaL/folders_to_move/macro_launch .
mv /WatChMaL/folders_to_move/motebooks .
mv /WatChMaL/folders_to_move/sweep_runs .


