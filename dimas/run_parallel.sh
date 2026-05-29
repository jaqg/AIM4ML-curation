#!/bin/bash

# Script for parallelizing the execution of MMPBSA scripts

TASK="python3 /datos_pool/mldata1/QMdatasets/OMol25/filter_omol25.py" 

INPUT_LIST="LISTA" 
PARALLEL="/opt/parallel-bash/bin/parallel --no-notice "   # GNU tool for parallel scriptexecution

# TEMP Directory
if [ ! -n "$PBS_ENVIRONMENT" ] ; then
   SCRATCH=/scratch
   NPROCS="6"
else 
   NPROCS=$(cat $PBS_NODEFILE|wc -l) 
fi

rm -f TASK.sh

NINPUT=$(cat $INPUT_LIST | wc -l)

for file in $(cat $INPUT_LIST)
do
  echo " $TASK  $file >  $SCRATCH/temp_${file}.log " >> TASK.sh
done
echo "Running parallel $TASK across  $NPROCS  procs ..."
cat TASK.sh | $PARALLEL -j $NPROCS

rm -f $SCRATCH/temp_*log

