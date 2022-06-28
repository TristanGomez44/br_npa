case $1 in
  "br_npa_low_res")
    python trainVal.py -c config_cub.config --model_id br_npa_low_res     --attention True --br_npa True --stride_lay3 2 --stride_lay4 2 --max_batch_size 130 --optuna False --attention_metrics Del
    ;;
  "br_npa_high_res_distill")
    python trainVal.py -c config_cub.config --model_id br_npa_high_res_distill   --attention True --br_npa True --max_batch_size 60 --val_batch_size 100 --master_net True --m_model_id clusRed --optuna False --attention_metrics Del
    ;;
  "b_cnn")
    python trainVal.py -c config_cub.config --model_id b_cnn      --attention True   --stride_lay3 2 --stride_lay4 2 --max_batch_size 130 --optuna False --attention_metrics Del
    ;;
  "cnn")
    python trainVal.py -c config_cub.config --model_id cnn      --stride_lay3 2 --stride_lay4 2 --max_batch_size 130 --optuna False --attention_metrics Del
    ;;
  "*")
    echo "no such model"
    ;;
esac
