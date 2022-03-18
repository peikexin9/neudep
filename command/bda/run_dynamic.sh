#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: ./run_dynamic.sh <binary>"
    echo "Environment variables:"
    echo "  QUICK_SAMPLE_LEVEL: speedup level for large binaries, higher means quicker but less accurate (default: 1)"
    echo "  SAMPLES: number of samples (default: 10000)"
    echo "  SBARE_CONFIG: the path of a given .sabre.rc (default is $HOME/playground/bda/.sabre.rc)"
    echo "  SHOW_DETAILS: show all dependencies BDA found (default: false)"
    exit 1
fi

QUICK_SAMPLE_LEVEL=${QUICK_SAMPLE_LEVEL:-"1"}
SAMPLES=${SAMPLES:-"10000"}
SABRE_CONFIG=${SABRE_CONFIG:-"$HOME/playground/bda/.sabre.rc"}
[[ -z "${SHOW_DETAILS}" ]] && DEPOPT="" || DEPOPT="--show-details"

QEMU_TOOL="$HOME/qemu-trace/run.sh"

ARGV1=$1
BINARY_NAME=$(basename $(echo "${ARGV1/\//}"))
BINARY_DIR=$BINARY_NAME

# step 1. get dynamic information
DYNAMIC_DIR=$(realpath $PWD)/$BINARY_NAME.dynamic
rm -rf $DYNAMIC_DIR
cp -r $BINARY_DIR $DYNAMIC_DIR

COMMANDS_PATH=$DYNAMIC_DIR/commands.txt
if [ -f "$COMMANDS_PATH" ]; then
    echo "$COMMANDS_PATH exists"
else
    echo "$COMMANDS_PATH does not exist"
    exit 1
fi

COUNTER=0
cd $DYNAMIC_DIR
while IFS= read -r line; do
    WORKDIR=trace_$COUNTER $QEMU_TOOL $line
    let COUNTER++
done < $COMMANDS_PATH
cd ..

# step 2. generate .sabre.rc
DYNAMIC_GENERATOR=$(realpath $PWD)/.dynamic_generator.py
$DYNAMIC_GENERATOR $DYNAMIC_DIR $BINARY_NAME $COUNTER $QUICK_SAMPLE_LEVEL $SABRE_CONFIG


# step 3. invoke BDA
WORKDIR=$BINARY_NAME.bda

rm -rf $WORKDIR
cp -r $DYNAMIC_DIR $WORKDIR
cd $WORKDIR

# step 3.1: build the basic control flow graph
rexe $BINARY_NAME
# step 3.2: sample for $SAMPLES rounds
rexe -t $SAMPLES $BINARY_NAME
# step 3.3: analyze dependency
rdep -d dynamic.dep $DEPOPT $BINARY_NAME
