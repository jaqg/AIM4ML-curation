#!/bin/bash

# Reorders an XYZ molecule using the AMBER PREP convention.
# Based on Dimas's xyz_reord.sh; see inline change tag for modifications.

BABEL_DIR="/usr/bin/"
AMBERHOME="/opt/amber23"
ANTECHAMBER=$AMBERHOME/bin/antechamber

if [ $# -eq "0" ]
then
  echo "Usage: xyz_reord_qm40.sh file.xyz"
  exit
fi

xyz=$1
WORKDIR=$PWD

TT=$(date +%N)
TMPDIR="/dev/shm/temp_${TT}"
if [ -e $TMPDIR ]
then
  rm -f -r $TMPDIR
fi
mkdir $TMPDIR

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
# [2026-05-05] jaqg: awk now strips trailing digits and converts to proper case, so CL3 -> Cl instead of CL3 -> C
grep -v DUMM temp_reordxyz.prepc | sed '1,7d' | sed -n "1,${NAT}p" | awk '{name=$2; gsub(/[0-9]+$/, "", name); printf(" %s %f %f %f \n", toupper(substr(name,1,1)) tolower(substr(name,2)), $5, $6, $7)}' >> temp_reordxyz.xyz

NCHECK=$(cat temp_reordxyz.xyz | wc -l)

cd $WORKDIR

if [ ${NCHECK} -eq ${NLINES} ]
then
  mv -f $TMPDIR/temp_reordxyz.xyz $xyz
else
  mv ${xyz} ${xyz}_ORIG
fi

rm -r -f $TMPDIR
