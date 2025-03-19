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
FILES=("mpc_data_generator.py")
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

# check data status
if [ ! -d "$DATA_DIR"/"$TASK"/initial_dataset/ ]; then
    echo -e "\e[1;31m[Info]\e[0m No initial dataset found."
    exit 1
else
    # check if there is only one file in the initial dataset
    FILE_COUNT=$(find "$DATA_DIR"/"$TASK"/initial_dataset/ -type f | wc -l)
    if [ "$FILE_COUNT" -eq 0 ]; then
        echo -e "\e[1;31m[Error]\e[0m No files found in the initial dataset."
        exit 1
    elif [ "$FILE_COUNT" -gt 1 ]; then
        echo -e "\e[1;31m[Error]\e[0m More than one file found in the initial dataset."
        exit 1
    fi
    INIT_DATA_PATH="$DATA_DIR"/"$TASK"/initial_dataset/$(ls "$DATA_DIR"/"$TASK"/initial_dataset/ | head -n 1)
    echo -e "\e[1;34m[Info]\e[0m Initial dataset founded: $INIT_DATA_PATH"
fi

if [ ! -d "$DATA_DIR"/"$TASK"/reference_repository/ ]; then
    echo -e "\e[1;31m[Info]\e[0m No reference repository found."
    exit 1
else
    FILE_COUNT=$(find "$DATA_DIR"/"$TASK"/reference_repository/ -type f | wc -l)
    if [ "$FILE_COUNT" -eq 0 ]; then
        echo -e "\e[1;31m[Error]\e[0m No files found in the reference_repository."
        exit 1
    elif [ "$FILE_COUNT" -gt 1 ]; then
        echo -e "\e[1;31m[Error]\e[0m More than one file found in the reference_repository."
        exit 1
    fi
    REF_REPO_PATH="$DATA_DIR"/"$TASK"/reference_repository/$(ls "$DATA_DIR"/"$TASK"/reference_repository/ | head -n 1)
    echo -e "\e[1;34m[Info]\e[0m Reference repository founded: $REF_REPO_PATH"
fi

if [ ! -d "$DATA_DIR"/"$TASK"/tracking_dataset/ ]; then
    mkdir -p "$DATA_DIR"/"$TASK"/tracking_dataset/
    TRACK_FILE_COUNT=0
    TRACK_DATA_PATHS=""
    echo -e "\e[1;31m[Info]\e[0m No tracking dataset founded. Train initial dynamics model."
else
    TRACK_FILE_COUNT=$(find "$DATA_DIR"/"$TASK"/tracking_dataset/ -type f | wc -l)
    if [ "$TRACK_FILE_COUNT" -eq 0 ]; then
        TRACK_DATA_PATHS=""
        echo -e "\e[1;34m[Info]\e[0m No tracking dataset founded. Train initial dynamics model."
    else
        TRACK_DATA_PATHS=$(find "$DATA_DIR"/"$TASK"/tracking_dataset/ -type f)
        echo -e "\e[1;34m[Info]\e[0m Tracking dataset founded: $TRACK_DATA_PATHS"
    fi
fi
echo
sleep 1

# get hyperparameters for the current training iteration
echo -e "\e[1;34m[Info]\e[0m Training the dynamics model..."
source configs/"$TASK".txt
LR_VAR="ITER_${TRACK_FILE_COUNT}_LR"
EPOCH_VAR="ITER_${TRACK_FILE_COUNT}_EPOCH"
ENCODE_DIM_VAR="ITER_${TRACK_FILE_COUNT}_ENCODE_DIM"
TRAJ_LEN_VAR="TRAJ_LEN"

LR=${!LR_VAR}
EPOCH=${!EPOCH_VAR}
ENCODE_DIM=${!ENCODE_DIM_VAR}
TRAJ_LEN=${!TRAJ_LEN_VAR}

if [ -z "$LR" ] || [ -z "$EPOCH" ] || [ -z "$ENCODE_DIM" ]; then
    echo -e "\e[1;31m[Error]\e[0m Hyperparameters not found for the current iteration."
    exit 1
fi

# train the dynamics model
if [ "$TRACK_FILE_COUNT" -eq 0 ]; then
    TRAIN_CMD="python script/runner/train.py --task "$TASK" --data_path "$INIT_DATA_PATH" \
            --encode_dim "$ENCODE_DIM" --traj_len "$TRAJ_LEN" --epoch "$EPOCH" --lr "$LR""
else
    TRAIN_CMD="python script/runner/train.py --task "$TASK" --data_path "$INIT_DATA_PATH" \
            --encode_dim "$ENCODE_DIM" --traj_len "$TRAJ_LEN" --epoch "$EPOCH" --lr "$LR" --track_data_paths "$TRACK_DATA_PATHS""
fi
if ! $TRAIN_CMD; then
    echo -e "\e[1;31m[Error]\e[0m Failed to train the dynamics model."
    exit 1
fi
echo -e "\e[1;34m[Info]\e[0m Finished training the dynamics model."
echo
sleep 1

# load lateset Koopman training log for the current iteration
MATCHED_DIRS=$(ls "$CURRENT_DIR"/logs/"$TASK"/ | grep Iter"$TRACK_FILE_COUNT" | sort -t '_' -k2,2r)
LATEST_DIR=$(echo "$MATCHED_DIRS" | head -n 1)
KOOPMAN_DIR="$CURRENT_DIR"/logs/"$TASK"/"$LATEST_DIR"
echo -e "\e[1;34m[Info]\e[0m Loading the latest Koopman training log: $KOOPMAN_DIR"
echo
sleep 1

# generate tracking dataset
cd "$ISAAC_DIR"
echo -e "\e[1;34m[Info]\e[0m Generating tracking dataset, Koopamn path: $KOOPMAN_DIR, reference path: $REF_REPO_PATH..."
if [ ! -d "$DATA_DIR"/"$TASK"/tracking_dataset/ ]; then
    mkdir -p "$DATA_DIR"/"$TASK"/tracking_dataset/
fi
if ! ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/mpc_data_generator.py \
            --headless --video --task "$TASK_ISAAC_PLAY" \
            --load_run "$KOOPMAN_DIR" \
            --ref "$REF_REPO_PATH" \
            --data_dir "$DATA_DIR"/"$TASK"/tracking_dataset/; then
    echo -e "\e[1;31m[Error]\e[0m Failed to generate reference repository. Iteration $TRACK_FILE_COUNT failed."
    exit 1
fi
echo -e "\e[1;34m[Info]\e[0m Finished generating tracking dataset and saved to "$DATA_DIR"/"$TASK"/tracking_dataset/. Iteration $TRACK_FILE_COUNT finished."