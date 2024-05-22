#!/usr/bin/env bash
MODEL_NAME="mfrl_test"
SERVER_IP="tcp://server-ip:5555"
MAX_EP=20
MAX_PROC=100
REAL_EVAL=200

# Pretraining
nice -n 15 python3 pretrain_mfrl_model.py -robot -mn "mfrl_pretrain" -tb -ep 40000

# Initialization
nice -n 15 python3 run_mfrl.py -robot -tb -ep 0 -mn $MODEL_NAME -ts 0 -nb 800 --test_every 20 -zip $SERVER_IP -ppoent 0.1 -ppoclip 0.2 -rbias 0.1 -ral 2 -ril 2 --bs_est 250 -r $REAL_EVAL
nice -n 15 python3 run_mfrl.py -robot -tb -ep 0 -mn $MODEL_NAME -ts 1 -nb 800 --test_every 20 -zip $SERVER_IP -ppoent 0.1 -ppoclip 0.2 -rbias 0.1 -ral 2 -ril 2 --bs_est 250 -r $REAL_EVAL

# Run MFRL
# nice -n 15 python3 run_mfrl.py -robot -tb -ep 5 -mn $MODEL_NAME -l -ld "../logs/mfrl/"$MODEL_NAME -lep 75 -ts 3 -nb 800 --test_every 20 -zip $SERVER_IP -ppoent 0.1 -ppoclip 0.2 -rbias 0.1 -ral 2 -ril 2 --bs_est 250 -r $REAL_EVAL
# nice -n 15 python3 run_mfrl.py -robot -tb -ep $MAX_EP -mn $MODEL_NAME -l -ld "../logs/mfrl/"$MODEL_NAME -lep 200 -ts 2 -nb 800 --test_every 20 -zip $SERVER_IP -ppoent 0.1 -ppoclip 0.2 -rbias 0.1 -ral 2 -ril 2 --bs_est 250 -r $REAL_EVAL
for (( k = 0; k < MAX_PROC; ++k )); do
nice -n 15 python3 run_mfrl.py -robot -tb -ep $MAX_EP -mn $MODEL_NAME -l -ld "../logs/mfrl/"$MODEL_NAME -lep $((k*MAX_EP)) -ts 3 -nb 800 --test_every 20 -zip $SERVER_IP -ppoent 0.1 -ppoclip 0.2 -rbias 0.1 -ral 2 -ril 2 --bs_est 250 -r $REAL_EVAL
  if (( $(((k+1)*MAX_EP)) % $REAL_EVAL == 0))
  then
  nice -n 15 python3 run_mfrl.py -robot -tb -ep $MAX_EP -mn $MODEL_NAME -l -ld "../logs/mfrl/"$MODEL_NAME -lep $(((k+1)*MAX_EP)) -ts 2 -nb 800 --test_every 20 -zip $SERVER_IP -ppoent 0.1 -ppoclip 0.2 -rbias 0.1 -ral 2 -ril 2 --bs_est 250 -r $REAL_EVAL
  fi
done
