#!/bin/bash

# This scripts reorders an XYZ molecule using the AMBER PREP convention

BABEL_DIR="/usr/bin/"
AMBERHOME="/opt/amber23"
ANTECHAMBER=$AMBERHOME/bin/antechamber 

#
if [ $# -eq "0" ]
then
  echo "Usage: xyz_reord.sh  file.xyz "
  exit 
fi

xyz=$1
WORKDIR=$PWD

# TEMP Directories
TT=$(date +%N)
TMPDIR="/dev/shm/temp_${TT}"
if [ -e $TMPDIR ]
then
  rm -f -r $TMPDIR
fi
mkdir $TMPDIR

echo "Reordering $xyz"

cp ${xyz} $TMPDIR
cd $TMPDIR

NAT=$(head -1 $xyz)
let "NLINES=$NAT+2"

$BABEL_DIR/obabel -ixyz $xyz -opdb -O temp_reordxyz.pdb  &> /dev/null
sed -i 's/HETATM/ATOM  /' temp_reordxyz.pdb
grep 'ATOM  ' temp_reordxyz.pdb >temp_reordxyz; mv temp_reordxyz temp_reordxyz.pdb 
$ANTECHAMBER -fi pdb -i temp_reordxyz.pdb -du no -an yes -fo prepc -o temp_reordxyz.prepc -pf yes &> /dev/null

echo "$NAT" > temp_reordxyz.xyz
echo "Reordered coords" >> temp_reordxyz.xyz
#  grep -v DUMM $TMPDIR/temp_reordxyz.prepc | sed '1,7d' | sed -n "1,${NAT}p" | awk '{gsub(/[0-9a-z]/, "", $2); printf(" %s %f %f %f \n",$2,$5,$6,$7)}' >> $TMPDIR/temp_reordxyz.xyz 
#  WARNING: awk prints only first character in atom symbol (GOOD for CHONS)
grep -v DUMM temp_reordxyz.prepc | sed '1,7d' | sed -n "1,${NAT}p" | awk '{printf(" %s %f %f %f \n",substr($2,1,1),$5,$6,$7)}' >> temp_reordxyz.xyz 

NCHECK=$(cat temp_reordxyz.xyz| wc -l)
   
cd  $WORKDIR

if [ ${NCHECK} -eq ${NLINES} ]
then 
   mv -f $TMPDIR/temp_reordxyz.xyz $xyz
else
   mv ${xyz} ${xyz}_ORIG
fi

rm -r -f $TMPDIR 

