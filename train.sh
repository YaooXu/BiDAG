# FB15k-237
CUDA_VISIBLE_DEVICES=6 python ./main.py --cuda --do_train --do_valid --do_test --data_path data/FB15k-237-betae -n 128 -b 2048 -d 400 --num_bi_dir_calibrates 3  -g 30  -lr 0.0002 --max_steps 500001 --cpu_num 10 --geo dagnn --valid_steps 10000 --tasks 1p.2p.3p.2i.3i.ip.pi.2u.up

# FB15k
CUDA_VISIBLE_DEVICES=7 python ./main.py --cuda --do_train --do_valid --do_test --data_path data/FB15k-betae -n 128 -b 2048 -d 400 --num_bi_dir_calibrates 3  -g 30  -lr 0.0003 --max_steps 500001 --cpu_num 10 --geo dagnn --valid_steps 10000 --tasks 1p.2p.3p.2i.3i.ip.pi.2u.up

# NELL
CUDA_VISIBLE_DEVICES=9 python ./main.py --cuda --do_train --do_valid --do_test --data_path data/NELL-betae -n 128 -b 512 -d 400 --num_bi_dir_calibrates 2  -g 24  -lr 0.0002 --max_steps 500001 --cpu_num 10 --geo dagnn --valid_steps 10000 --tasks 1p.2p.3p.2i.3i.ip.pi.2u.up


# record all layer mrrs
CUDA_VISIBLE_DEVICES=9 python ./main.py --cuda --fp16 --calc_all_layers_mrr --do_train --do_valid --do_test --data_path data/FB15k-237-betae -n 128 -b 512 -d 400 --num_bi_dir_calibrates 3  -g 30  -lr 0.0002 --max_steps 500001 --cpu_num 10 --geo dagnn --valid_steps 10000  --tasks 1p.2p.3p.2i.3i.ip.pi.2u.up
