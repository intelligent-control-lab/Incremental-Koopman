#!/bin/bash

# set the path to the IsaacLab directory
ISAAC_DIR=$1
TASK=$2
KOOPMAN_DIR=$3
REF_REPO_PATH=$4
if [[ -z "$ISAAC_DIR" ]]; then
    echo -e "\e[1;31m[Info]\e[0m Please provide the path to the IsaacLab directory."
    exit 1
else
    ISAAC_DIR=$(realpath "$ISAAC_DIR")
fi
if [[ -z "$TASK" ]]; then
    echo -e "\e[1;31m[Info]\e[0m Please provide the specific task name"
    exit 1
fi
if [[ -z "$KOOPMAN_DIR" ]]; then
    echo -e "\e[1;31m[Info]\e[0m Please provide the path to the Koopman model directory."
    exit 1
else
    KOOPMAN_DIR=$(realpath "$KOOPMAN_DIR")
fi
if [[ -z "$REF_REPO_PATH" ]]; then
    echo -e "\e[1;31m[Info]\e[0m Please provide the path to the reference repository."
    exit 1
else
    REF_REPO_PATH=$(realpath "$REF_REPO_PATH")
fi

case $TASK in
    g1_flat)
        TASK_ISAAC_TRAIN="Isaac-Velocity-Flat-G1-v0"
        TASK_ISAAC_PLAY="Isaac-Velocity-Flat-G1-Play-v0"
        ;;
    h1_flat)
        TASK_ISAAC_TRAIN="Isaac-Velocity-Flat-H1-v0"
        TASK_ISAAC_PLAY="Isaac-Velocity-Flat-H1-Play-v0"
        ;;
    unitree_go2_flat)
        TASK_ISAAC_TRAIN="Isaac-Velocity-Flat-Unitree-Go2-v0"
        TASK_ISAAC_PLAY="Isaac-Velocity-Flat-Unitree-Go2-Play-v0"
        ;;
    unitree_a1_flat)
        TASK_ISAAC_TRAIN="Isaac-Velocity-Flat-Unitree-A1-v0"
        TASK_ISAAC_PLAY="Isaac-Velocity-Flat-Unitree-A1-Play-v0"
        ;;
    anymal_d_flat)
        TASK_ISAAC_TRAIN="Isaac-Velocity-Flat-Anymal-D-v0"
        TASK_ISAAC_PLAY="Isaac-Velocity-Flat-Anymal-D-Play-v0"
        ;;
    *)
        echo -e "\e[1;31m[Info]\e[0m Invalid task name."
        exit 1
        ;;
esac

# save current path
CURRENT_DIR=$(pwd)

# copy the necessary code to the IsaacLab directory if not exists
FILES=("mpc.py")
SRC_DIR="$CURRENT_DIR/code_for_isaaclab"
DEST_DIR="$ISAAC_DIR/scripts/reinforcement_learning/rsl_rl"
for FILE in "${FILES[@]}"; do
    if [[ -f "$DEST_DIR/$FILE" ]]; then
        echo "File $FILE exists in $DEST_DIR, skipping..."
    else
        echo "Copying $FILE to $DEST_DIR..."
        cp "$SRC_DIR/$FILE" "$DEST_DIR/"
        echo -e "\e[1;34m[Info]\e[0m Copied the necessary code to the IsaacLab directory."
    fi
done
echo
sleep 1

# generate videos for specific koopman model and reference repository
cd "$ISAAC_DIR"
if ! ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/mpc.py \
            --headless --video --task "$TASK_ISAAC_PLAY" \
            --load_run "$KOOPMAN_DIR" \
            --ref "$REF_REPO_PATH" ; then
    echo -e "\e[1;31m[Error]\e[0m Failed to generate videos."
    exit 1
fi
echo -e "\e[1;34m[Info]\e[0m Finished generating videos for the Koopman model: "$KOOPMAN_DIR" and reference repository: "$REF_REPO_PATH"."