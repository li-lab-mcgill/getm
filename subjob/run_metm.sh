# python main_multi_etm_sep.py --bow_norm=0 --epochs=8 --lr=0.01 --batch_size=100 --save_path="Multi_sup_fixed/results_m640c332_topic128"\
#  --vocab_size1=640 --vocab_size2=332 --data_path="Supervised/drug640_cond332" --num_topics=128 --rho_size=128 --emb_size=128 --t_hidden_size=64 --enc_drop=0.0 \
#  --train_embeddings1=0 --embedding1="drug_emb.npy" --train_embeddings2=0 --embedding2="cond_emb.npy" --rho_fixed1=1 --rho_fixed2=1\
#   --lstm_hidden_size=10 --e2e=0 --num_classes=8 --pred_nlayer=1 --predcoef=2

# python main_multi_etm_sep.py --bow_norm=0 --epochs=8 --lr=0.01 --batch_size=100 --save_path="Multi_sup_fixed/results_m605c284_topic128"\
#  --vocab_size1=605 --vocab_size2=284 --data_path="Supervised/drug605_cond284" --num_topics=128 --rho_size=128 --emb_size=128 --t_hidden_size=64 --enc_drop=0.0 \
#  --train_embeddings1=0 --embedding1="drug_emb.npy" --train_embeddings2=0 --embedding2="cond_emb.npy" --rho_fixed1=1 --rho_fixed2=1\
#   --lstm_hidden_size=10 --e2e=0 --num_classes=8 --pred_nlayer=1 --predcoef=2

# python main_multi_etm_sep.py --bow_norm=0 --epochs=8 --lr=0.01 --batch_size=100 --save_path="Multi_sup_fixed/results_m674c341_topic128"\
#  --vocab_size1=674 --vocab_size2=341 --data_path="Supervised/drug674_cond341" --num_topics=128 --rho_size=128 --emb_size=128 --t_hidden_size=64 --enc_drop=0.0 \
#  --train_embeddings1=0 --embedding1="drug_emb.npy" --train_embeddings2=0 --embedding2="cond_emb.npy" --rho_fixed1=1 --rho_fixed2=1\
#   --lstm_hidden_size=10 --e2e=0 --num_classes=8 --pred_nlayer=1 --predcoef=2

# python main_multi_etm_sep.py --bow_norm=0 --epochs=8 --lr=0.01 --batch_size=100 --save_path="Multi_sup_fixed/results_m661c336_topic128"\
#  --vocab_size1=661 --vocab_size2=336 --data_path="Supervised/drug661_cond336" --num_topics=128 --rho_size=128 --emb_size=128 --t_hidden_size=64 --enc_drop=0.0 \
#  --train_embeddings1=0 --embedding1="drug_emb.npy" --train_embeddings2=0 --embedding2="cond_emb.npy" --rho_fixed1=1 --rho_fixed2=1\
#   --lstm_hidden_size=10 --e2e=0 --num_classes=8 --pred_nlayer=1 --predcoef=2

# ###########################################################################################################################################################
# python main_multi_etm_sep.py --bow_norm=0 --epochs=8 --lr=0.01 --batch_size=100 --save_path="Multi_sup_fixed/results_m671c342_topic128"\
#  --vocab_size1=671 --vocab_size2=342 --data_path="Supervised/drug671_cond342" --num_topics=128 --rho_size=128 --emb_size=128 --t_hidden_size=64 --enc_drop=0.0 \
#  --train_embeddings1=0 --embedding1="drug_emb.npy" --train_embeddings2=0 --embedding2="cond_emb.npy" --rho_fixed1=1 --rho_fixed2=1\
#   --lstm_hidden_size=10 --e2e=0 --num_classes=8 --pred_nlayer=1 --predcoef=2

# python main_multi_etm_sep.py --bow_norm=0 --epochs=8 --lr=0.01 --batch_size=100 --save_path="Multi_sup_fixed/results_m670c337_topic128"\
#  --vocab_size1=670 --vocab_size2=337 --data_path="Supervised/drug670_cond337" --num_topics=128 --rho_size=128 --emb_size=128 --t_hidden_size=64 --enc_drop=0.0 \
#  --train_embeddings1=0 --embedding1="drug_emb.npy" --train_embeddings2=0 --embedding2="cond_emb.npy" --rho_fixed1=1 --rho_fixed2=1\
#   --lstm_hidden_size=10 --e2e=0 --num_classes=8 --pred_nlayer=1 --predcoef=2

# python main_multi_etm_sep.py --bow_norm=0 --epochs=8 --lr=0.01 --batch_size=100 --save_path="Multi_sup_fixed/results_m662c338_topic128"\
#  --vocab_size1=662 --vocab_size2=338 --data_path="Supervised/drug662_cond338" --num_topics=128 --rho_size=128 --emb_size=128 --t_hidden_size=64 --enc_drop=0.0 \
#  --train_embeddings1=0 --embedding1="drug_emb.npy" --train_embeddings2=0 --embedding2="cond_emb.npy" --rho_fixed1=1 --rho_fixed2=1\
#   --lstm_hidden_size=10 --e2e=0 --num_classes=8 --pred_nlayer=1 --predcoef=2

# python main_multi_etm_sep.py --bow_norm=0 --epochs=8 --lr=0.01 --batch_size=100 --save_path="Multi_sup_fixed/results_m665c348_topic128"\
#  --vocab_size1=665 --vocab_size2=348 --data_path="Supervised/drug665_cond348" --num_topics=128 --rho_size=128 --emb_size=128 --t_hidden_size=64 --enc_drop=0.0 \
#  --train_embeddings1=0 --embedding1="drug_emb.npy" --train_embeddings2=0 --embedding2="cond_emb.npy" --rho_fixed1=1 --rho_fixed2=1\
#   --lstm_hidden_size=10 --e2e=0 --num_classes=8 --pred_nlayer=1 --predcoef=2

for topic_num in 64 128 256 512
do
	python main_multi_etm_sep.py --bow_norm=0 --epochs=10 --lr=0.01 --batch_size=100 --save_path="acute2chronic_results/results_m579c322_topic${topic_num}"\
	 --vocab_size1=579 --vocab_size2=322 --data_path="Acute2Chronic/drug579_cond322" --num_topics=${topic_num} --rho_size=128 --emb_size=128 --t_hidden_size=64 --enc_drop=0.0 \
	 --train_embeddings1=0 --embedding1="drug_emb.npy" --train_embeddings2=0 --embedding2="code_emb.npy" --rho_fixed1=1 --rho_fixed2=1\
	  --lstm_hidden_size=10 --e2e=0 --num_classes=8 --pred_nlayer=1 --predcoef=2
done

# for topic_num in 64 128 256 512
# do
# 	python main_multi_etm_sep.py --bow_norm=0 --epochs=10 --lr=0.01 --batch_size=100 --save_path="acute2chronic_results/results_m680c389_topic${topic_num}"\
# 	 --vocab_size1=680 --vocab_size2=389 --data_path="Acute2Chronic/drug680_cond389" --num_topics=${topic_num} --rho_size=128 --emb_size=128 --t_hidden_size=64 --enc_drop=0.0 \
# 	 --train_embeddings1=0 --embedding1="drug_emb.npy" --train_embeddings2=0 --embedding2="code_emb.npy" --rho_fixed1=1 --rho_fixed2=1\
# 	  --lstm_hidden_size=10 --e2e=0 --num_classes=8 --pred_nlayer=1 --predcoef=2
# done

# for topic_num in 64 128 256 512
# do
# 	python main_multi_etm_sep.py --bow_norm=0 --epochs=10 --lr=0.01 --batch_size=100 --save_path="acute2chronic_results/results_m680c351_topic${topic_num}"\
# 	 --vocab_size1=680 --vocab_size2=351 --data_path="Acute2Chronic/drug680_cond351" --num_topics=${topic_num} --rho_size=128 --emb_size=128 --t_hidden_size=64 --enc_drop=0.0 \
# 	 --train_embeddings1=0 --embedding1="drug_emb.npy" --train_embeddings2=0 --embedding2="code_emb.npy" --rho_fixed1=1 --rho_fixed2=1\
# 	  --lstm_hidden_size=10 --e2e=0 --num_classes=8 --pred_nlayer=1 --predcoef=2
# done

# for topic_num in 64 128 256 512
# do
# 	python main_multi_etm_sep.py --bow_norm=0 --epochs=10 --lr=0.01 --batch_size=100 --save_path="acute2chronic_results/results_m579c389_topic${topic_num}"\
# 	 --vocab_size1=579 --vocab_size2=389 --data_path="Acute2Chronic/drug579_cond389" --num_topics=${topic_num} --rho_size=128 --emb_size=128 --t_hidden_size=64 --enc_drop=0.0 \
# 	 --train_embeddings1=0 --embedding1="drug_emb.npy" --train_embeddings2=0 --embedding2="code_emb.npy" --rho_fixed1=1 --rho_fixed2=1\
# 	  --lstm_hidden_size=10 --e2e=0 --num_classes=8 --pred_nlayer=1 --predcoef=2
# done

# for topic_num in 64 128 256 512
# do
# 	python main_multi_etm_sep.py --bow_norm=0 --epochs=10 --lr=0.01 --batch_size=100 --save_path="acute2chronic_results/results_m579c351_topic${topic_num}"\
# 	 --vocab_size1=579 --vocab_size2=351 --data_path="Acute2Chronic/drug579_cond351" --num_topics=${topic_num} --rho_size=128 --emb_size=128 --t_hidden_size=64 --enc_drop=0.0 \
# 	 --train_embeddings1=0 --embedding1="drug_emb.npy" --train_embeddings2=0 --embedding2="code_emb.npy" --rho_fixed1=1 --rho_fixed2=1\
# 	  --lstm_hidden_size=10 --e2e=0 --num_classes=8 --pred_nlayer=1 --predcoef=2
# done

# for topic_num in 15 50 75 128 256
# do
# 	python main.py --bow_norm=0 --epochs=10 --lr=0.01 --batch_size=100 --save_path="Multi_sup/results_flattened_m802c444_fixed_topic${topic_num}"\
# 	 --vocab_size=1246 --data_path="Supervised/drug802_cond444" --num_topics=${topic_num} --rho_size=128 --emb_size=128 --t_hidden_size=64 --enc_drop=0.0 \
# 	 --train_embeddings=1 --embedding="both_emb.npy" --rho_fixed=1\
# 	  --lstm_hidden_size=10 --e2e=0 --num_classes=8 --pred_nlayer=1 --predcoef=10
# done

# for topic_num in 15 50 75 128 256
# do
# 	python main.py --bow_norm=0 --epochs=10 --lr=0.01 --batch_size=100 --save_path="Multi_sup/results_flattened_m680c405_fixed_topic${topic_num}"\
# 	 --vocab_size=1085 --data_path="Supervised/drug680_cond405" --num_topics=${topic_num} --rho_size=128 --emb_size=128 --t_hidden_size=64 --enc_drop=0.0 \
# 	 --train_embeddings=1 --embedding="both_emb.npy" --rho_fixed=1\
# 	  --lstm_hidden_size=10 --e2e=0 --num_classes=8 --pred_nlayer=1 --predcoef=10
# done

# for topic_num in 15 50 75 128 256
# do
# 	python main.py --bow_norm=0 --epochs=10 --lr=0.01 --batch_size=100 --save_path="Multi_sup/results_flattened_m680c394_fixed_topic${topic_num}"\
# 	 --vocab_size=1074 --data_path="Supervised/drug680_cond394" --num_topics=${topic_num} --rho_size=128 --emb_size=128 --t_hidden_size=64 --enc_drop=0.0 \
# 	 --train_embeddings=1 --embedding="both_emb.npy" --rho_fixed=1\
# 	  --lstm_hidden_size=10 --e2e=0 --num_classes=8 --pred_nlayer=1 --predcoef=10
# done

# for topic_num in 15 50 75 128 256
# do
# 	python main.py --bow_norm=0 --epochs=10 --lr=0.01 --batch_size=100 --save_path="Multi_sup/results_flattened_m680c372_fixed_topic${topic_num}"\
# 	 --vocab_size=1052 --data_path="Supervised/drug680_cond372" --num_topics=${topic_num} --rho_size=128 --emb_size=128 --t_hidden_size=64 --enc_drop=0.0 \
# 	 --train_embeddings=1 --embedding="both_emb.npy" --rho_fixed=1\
# 	  --lstm_hidden_size=10 --e2e=0 --num_classes=8 --pred_nlayer=1 --predcoef=10
# done









