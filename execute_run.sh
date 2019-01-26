export BERT_BASE_DIR=/Users/porter/work/research/multi_cased_L-12_H-768_A-12
#export GLUE_DIR=/Users/porter/work/research/glue_data
export FLIGHT_INTENT_DIR=/Users/porter/work/research/query_intension/flight_intent/fine_tune_bert/

python run_classifier.py \
  --task_name=BinarySentence \
  --do_train=true \
  --do_eval=true \
  --data_dir=$FLIGHT_INTENT_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --do_lower_case=False \
  --output_dir=$FLIGHT_INTENT_DIR/fine_tuned_model/
