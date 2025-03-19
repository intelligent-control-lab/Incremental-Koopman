#!/bin/bash

# set the path to the IsaacLab directory
ISAAC_DIR=$1
TASK=$2
DATA_DIR=$3
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
if [[ -z "$DATA_DIR" ]]; then
    echo -e "\e[1;31m[Info]\e[0m Please provide the path to the data directory."
    exit 1
else
    DATA_DIR=$(realpath "$DATA_DIR")
    if [ ! -d "$DATA_DIR" ]; then
        echo -e "\e[1;34m[Info]\e[0m Creating the data directory..."
        mkdir -p "$DATA_DIR"
    fi
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

# copy the necessary code to the IsaacLab directory
FILES=("initial_data_generator.py" "mpc_data_generator.py" "mpc.py")
SRC_DIR="$CURRENT_DIR/code_for_isaaclab"
DEST_DIR="$ISAAC_DIR/scripts/reinforcement_learning/rsl_rl"
for FILE in "${FILES[@]}"; do
    if [[ -f "$DEST_DIR/$FILE" ]]; then
        echo "File $FILE exists in $DEST_DIR, skipping..."
    else
        echo "Copying $FILE to $DEST_DIR..."
        cp "$SRC_DIR/$FILE" "$DEST_DIR/"
    fi
done
echo -e "\e[1;34m[Info]\e[0m Copied the necessary code to the IsaacLab directory."
echo
sleep 1

# train reinforcement learning data collector
cd "$ISAAC_DIR"
echo -e "\e[1;34m[Info]\e[0m Training reinforcement learning data collector..."
if ! ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --headless --task $TASK_ISAAC_TRAIN; then
    echo -e "\e[1;31m[Error]\e[0m Failed to train reinforcement learning data collector."
    exit 1
fi
echo -e "\e[1;34m[Info]\e[0m Finished training reinforcement learning data collector."
echo
sleep 1

# obtain the latest training log
TRAIN_LOG=$(ls "$ISAAC_DIR"/logs/rsl_rl/"$TASK" | head -n 1)
echo -e "\e[1;34m[Info]\e[0m Loading the latest training log: "$ISAAC_DIR"/logs/rsl_rl/"$TASK"/"$TRAIN_LOG""

# generate initial dataset
echo -e "\e[1;34m[Info]\e[0m Generating initial dataset..."
if [ ! -d "$DATA_DIR"/"$TASK"/initial_dataset/ ]; then
    mkdir -p "$DATA_DIR"/"$TASK"/initial_dataset/
fi
if ! ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/initial_data_generator.py \
              --headless --video --task "$TASK_ISAAC_PLAY" \
              --load_run "$TRAIN_LOG" \
              --max_iterations 20 --num_envs 30 --max_episode_len 100 \
              --collect_data \
              --data_dir "$DATA_DIR"/"$TASK"/initial_dataset/; then
    echo -e "\e[1;31m[Error]\e[0m Failed to generate initial dataset."
    exit 1
fi
echo -e "\e[1;34m[Info]\e[0m Finished generating initial dataset."
echo
sleep 1

# generate reference repository
echo -e "\e[1;34m[Info]\e[0m Generating reference repository..."
if [ ! -d "$DATA_DIR"/"$TASK"/reference_repository/ ]; then
    mkdir -p "$DATA_DIR"/"$TASK"/reference_repository/
fi
if ! ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/initial_data_generator.py \
              --headless --video --task "$TASK_ISAAC_PLAY" \
              --load_run "$TRAIN_LOG" \
              --max_iterations 1 --num_envs 30 --max_episode_len 500 \
              --collect_data \
              --data_dir "$DATA_DIR"/"$TASK"/reference_repository/ \
              --noise_scale 0.05; then
    echo -e "\e[1;31m[Error]\e[0m Failed to generate reference repository."
    exit 1
fi
echo -e "\e[1;34m[Info]\e[0m Finished generating reference repository."