echo =======================

seeds=( 51061 )

steps_list=(3)
thera_hidden_dims=(64)

for seed in "${seeds[@]}"
do
    for steps in "${steps_list[@]}"
    do
        for thera_dim in "${thera_hidden_dims[@]}"
        do
            echo "--- Iteration with Seed $seed, Steps $steps, Thera Hidden Dim $thera_dim ---"
            python -u train.py \
                --lr 1e-4 \
                --batch-size 16 \
                --epochs 60 \
                --temp 8  \
                --Dataset 'IEMOCAP' \
                --steps $steps \
                --thera_hidden_dim $thera_dim \
                --lambd 0.5 0.4 0.6 0.8 0.5 \
                --MOE_depth 4 \
                --seed $seed \
                --checkpoint_dir checkpoints | tee -a "sdt_iemo.txt"
        done
    done
done
