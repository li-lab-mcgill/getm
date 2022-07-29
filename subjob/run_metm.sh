
python main_multi_etm_sep.py --bow_norm=0 --epochs=10 --lr=0.01 --batch_size=100 --save_path="acute2chronic_results/results_m579c322_topic${topic_num}"\
 --vocab_size1=579 --vocab_size2=322 --data_path="Acute2Chronic/drug579_cond322" --num_topics=${topic_num} --rho_size=128 --emb_size=128 --t_hidden_size=64 --enc_drop=0.0 \
 --train_embeddings1=0 --embedding1="drug_emb.npy" --train_embeddings2=0 --embedding2="code_emb.npy" --rho_fixed1=1 --rho_fixed2=1\
  --lstm_hidden_size=10 --e2e=0 --num_classes=8 --pred_nlayer=1 --predcoef=2


