#!/usr/bin/env bash
MODEL_NAME="mfrl_test"
SERVER_IP="tcp://server-ip:5555"
MAX_EP=20
MAX_PROC=100

# # Pretraining
# nice -n 15 python3 pretrain_mfrl_model.py -mn "mfrl_pretrain" -tb -ep 40000

# Initialization
nice -n 15 python3 run_mfrl.py -tb -ep 0 -mn $MODEL_NAME -ts 0 -nb 100 --test_every 20 -zip $SERVER_IP -ppoent 0.1 -ppoclip 0.2 -rbias 0.1 -ral 2 -ril 2 --bs_est 250
nice -n 15 python3 run_mfrl.py -tb -ep 0 -mn $MODEL_NAME -ts 1 -nb 100 --test_every 20 -zip $SERVER_IP -ppoent 0.1 -ppoclip 0.2 -rbias 0.1 -ral 2 -ril 2 --bs_est 250

# Run MFRL
# nice -n 15 python3 run_mfrl.py -tb -ep 5 -mn $MODEL_NAME -l -ld "../logs/mfrl/"$MODEL_NAME -lep 0 -ts 3 -nb 800 --test_every 20 -zip $SERVER_IP -ppoent 0.1 -ppoclip 0.2 -rbias 0.1 -ral 2 -ril 2 --bs_est 250
for (( k = 0; k < MAX_PROC; ++k )); do
nice -n 15 python3 run_mfrl.py -tb -ep $MAX_EP -mn $MODEL_NAME -l -ld "../logs/mfrl/"$MODEL_NAME -lep $((k*MAX_EP)) -ts 3 -nb 800 --test_every 20 -zip $SERVER_IP -ppoent 0.1 -ppoclip 0.2 -rbias 0.1 -ral 2 -ril 2 --bs_est 250
done
