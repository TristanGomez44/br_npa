case $1 in
  "br_npa_low_res")
    python trainVal.py -c config_cub.config --model_id br_npa_low_res     --attention True --br_npa True --stride_lay3 2 --stride_lay4 2 --max_batch_size 130 --optuna False --attention_metrics Add
    ;;
  "br_npa_high_res_distill")
    python trainVal.py -c config_cub.config --model_id br_npa_high_res_distill   --attention True --br_npa True --optuna False --attention_metrics Add
    ;;
  "b_cnn")
    python trainVal.py -c config_cub.config --model_id b_cnn      --attention True   --stride_lay3 2 --stride_lay4 2 --optuna False --attention_metrics Add
    ;;
  "cnn")
    python trainVal.py -c config_cub.config --model_id cnn      --stride_lay3 2 --stride_lay4 2 --optuna False --attention_metrics Add
    ;;
  "*")
    echo "no such model"
    ;;
esac
