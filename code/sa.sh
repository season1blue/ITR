export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

python code/main.py \
--do_eval_steps  20000 \
--eval_stop_step 100000 \
--train_epoch 100 \
--aspect_epochs 5 \
--model_name 'llava' \
--task_type 'SA_15' \
--dataset_dir '../data/el/twitter2015/' \
--img_file '../data/el/twitter2015/image_feature.h5' \
--path_image '../data/el/ImgData/twitter2015/' \
--txtlog_dir 'log_dir/log_sa.log' \
--ICL_examples_num 1 \
--example_file ../data/el/twitter2015/example.json \
--pre_predict


# accelerate launch main.py \
# --do_eval_steps 10000 \
# --train_epoch 100 \
# --model_name 'llama' \
# --task_type 'SA_17' \
# --dataset_dir '../../data/twitter2017/' \
# --img_file '../../data/twitter2017/image_feature.h5' \
# --path_image '../../data/ImgData/twitter2017/' \
# --txtlog_dir 'log_sa.log' \
# --ICL_examples_num 1 \
# --example_file ../../data/twitter2017/example.json  
