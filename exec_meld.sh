echo =======================

seeds=(
65256
)

steps_list=(5)
thera_hidden_dims=(128)

for seed in "${seeds[@]}"
do
    for steps in "${steps_list[@]}"
    do
        for thera_dim in "${thera_hidden_dims[@]}"
        do
            echo "--- Iteration with Seed $seed, Steps $steps, Thera Hidden Dim $thera_dim ---"
            python -u train.py \
                --lr 5e-6 \
                --batch-size 16 \
                --epochs 30 \
                --temp 8  \
                --Dataset 'MELD' \
                --steps $steps \
                --thera_hidden_dim $thera_dim \
                --MOE_depth 3 \
                --lambd 0.6 0.7 0.8 1.0 0.4 \
                --seed $seed \
                --checkpoint_dir checkpoints | tee -a "sdt_meld.txt"
        done
    done
done
