#!/bin/bash
#BSUB -q HPC.S1.GPU.X795.sha
#BSUB -n 8
#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -gpu  "num=1:gmem=3000M"
python ./main.py
sleep 5h
