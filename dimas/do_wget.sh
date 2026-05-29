#!/bin/bash

while read line 
do

echo $line 
wget $(echo $line | awk '{print $1}') 

done < $1
