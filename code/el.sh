export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

accelerate launch main.py \
--do_eval_steps 20000 \
--train_epoch 10 \
--model_name 'llama' \
--task_type 'EL_MEL' \

