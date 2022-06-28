case $1 in
  "br_npa_low_res")
    python trainVal.py -c config_cub.config --model_id br_npa_low_res     --attention True --br_npa True --stride_lay3 2 --stride_lay4 2 --max_batch_size 130
    ;;
  "br_npa_high_res_distill")
    python trainVal.py -c config_cub.config --model_id br_npa_high_res_distill   --attention True --br_npa True --max_batch_size 60 --val_batch_size 100 --master_net True --m_model_id clusRed
    ;;
  "b_cnn")
    python trainVal.py -c config_cub.config --model_id b_cnn      --attention True   --stride_lay3 2 --stride_lay4 2 --max_batch_size 130
    ;;
  "cnn")
    python trainVal.py -c config_cub.config --model_id cnn      --stride_lay3 2 --stride_lay4 2 --max_batch_size 130
    ;;
  "br_npa_high_res_random_vectors_no_refining")
    python trainVal.py -c config_cub.config --model_id br_npa_high_res_rand_vec_no_ref     --attention True --br_npa True  --br_npa_randvec True  --br_npa_norefine True --optuna False --max_batch_size_single_pass 9 
    ;;
  "br_npa_high_res_random_vectors")
    python trainVal.py -c config_cub.config --model_id br_npa_high_res_rand_vec     --attention True --br_npa True  --br_npa_randvec True --optuna False --max_batch_size_single_pass 9 
    ;;
  "br_npa_high_res_no_refining")
    python trainVal.py -c config_cub.config --model_id br_npa_high_res_no_ref     --attention True --br_npa True  --br_npa_norefine True --optuna False --max_batch_size_single_pass 9 
    ;;
  "br_npa_high_res_auxiliary_heads")
    python trainVal.py -c config_cub.config --model_id br_npa_high_res_aux_heads     --attention True --br_npa True  --aux_on_masked True  --optuna False --max_batch_size_single_pass 12
    ;;
  "*")
    echo "no such model"
    ;;
esac
