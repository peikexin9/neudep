#!/bin/bash


if [ "$#" -ne 1 ]; then
    echo "Usage: ./run_static.sh <binary>"
    echo "Environment variables:"
    echo "  QUICK_SAMPLE_LEVEL: speedup level for large binaries, higher means quicker but less accurate (default: 1)"
    echo "  SAMPLES: number of samples (default: 10000)"
    echo "  SABRE_CONFIG: the path of a given .sabre.rc (default is $HOME/playground/bda/.sabre.rc)"
    echo "  GHIDRA_DIR: the directory of Ghidra (default: $HOME/ghidra)"
    exit 1
fi

QUICK_SAMPLE_LEVEL=${QUICK_SAMPLE_LEVEL:-"1"}
SAMPLES=${SAMPLES:-"10000"}
SRC_CONFIG_PATH=${SABRE_CONFIG:-"$HOME/playground/bda/.sabre.rc"}

ARGV1=$1
PROGRAM=$(basename $(echo "${ARGV1/\//}"))

PROGRAM_DIR="$HOME/playground/bda/$PROGRAM"
WORKING_DIR="$PROGRAM_DIR.bda"
GHIDRA_SCRIPT_PATH="$HOME/sabre/osprey_py/ghidra.sh"

rm -rf $WORKING_DIR
cp -r $PROGRAM_DIR $WORKING_DIR

cd $WORKING_DIR

# step 1. get ghidra information and jump table
JUMPTABLE_PATH="$WORKING_DIR/$PROGRAM.ghidra.jmp"
XALLOC_PATH="$WORKING_DIR/xalloc.inc"
CONFIG_PATH="$WORKING_DIR/.sabre.rc"
$GHIDRA_SCRIPT_PATH $PROGRAM
cp $SRC_CONFIG_PATH $CONFIG_PATH
echo "DumpIndirectTransfer: true" >> "$CONFIG_PATH"
cat "$JUMPTABLE_PATH" >> "$CONFIG_PATH"
if [ -f "$XALLOC_PATH" ]; then
    cat "$XALLOC_PATH" | xargs printf "XAlloc: %s 1\n" >> "$CONFIG_PATH"
fi

echo $PROGRAM
read -n 1 -s

# step 2. rexe several times
for i in {1..10}
do
    rexe $PROGRAM
done

echo $PROGRAM
read -n 1 -s

# step 3. remove ubg file and sample 10000 times
UBG_PATH="$WORKING_DIR/$PROGRAM.ubg"
rm $UBG_PATH
rexe -t $SAMPLES $PROGRAM

echo $PROGRAM
read -n 1 -s

# step 4. analyze the dependency
rdep $PROGRAM
