# provide arguments in this order
# 1. GPU
# 2. attack_version
# 3. n_alt_iters
# 4. z_optim
# 5. z_init
# 6. z_epsilon
# 7. u_optim
# 8. u_pgd_epochs (v2) / pgd_epochs (v3)
# 9. u_accumulate_best_replacements
# 10. u_rand_update_pgd
# 11. use_loss_smoothing
# 12. short_name
# 13. src_field
# 14. u_learning_rate (v3)
# 15. z_learning_rate (v3)
# 16. smoothing_param (v2/3)
# 17. dataset (sri/py150 or c2s/java-small)
# 18. vocab_to_use 
# 19. model_in
# 20. number of replacement tokens
# 21. exact_matches (1 or 0)

# DATASET_NAME="sri/py150"
# DATASET_NAME_SMALL="py150"
DATASET_NAME="c2s/java-small"
DATASET_NAME_SMALL="javasmall"
TRANSFORM_NAME="transforms.Combined"
MODEL_NAME="final-models/seq2seq/$DATASET_NAME/normal"
NUM_REPLACE=1500

# 1 site
#./experiments/attack_and_test_seq2seq.sh 1 2 1 true 1 1 true 1 false false false v2-1-z_o_1-pgd_1_no-$TRANSFORM_NAME-$DATASET_NAME_SMALL $TRANSFORM_NAME 0.5 0.5 0.01 $DATASET_NAME 1 $MODEL_NAME $NUM_REPLACE 1

#./experiments/attack_and_test_seq2seq.sh 1 2 3 true 1 1 true 3 false false false v2-2-z_o_1-pgd_3_no-$TRANSFORM_NAME-$DATASET_NAME_SMALL $TRANSFORM_NAME 0.5 0.5 0.01 $DATASET_NAME 1 $MODEL_NAME $NUM_REPLACE 1

#./experiments/attack_and_test_seq2seq.sh 1 2 10 true 1 1 true 10 false false false v2-3-z_o_1-pgd_10_no-$TRANSFORM_NAME-$DATASET_NAME_SMALL $TRANSFORM_NAME 0.5 0.5 0.01 $DATASET_NAME 1 $MODEL_NAME $NUM_REPLACE 1

#./experiments/attack_and_test_seq2seq.sh 1 2 20 true 1 1 true 20 false false false v2-4-z_o_1-pgd_20_no-$TRANSFORM_NAME-$DATASET_NAME_SMALL $TRANSFORM_NAME 0.5 0.5 0.01 $DATASET_NAME 1 $MODEL_NAME $NUM_REPLACE 1

# 5 sites
./experiments/attack_and_test_seq2seq.sh 0 2 3 true 1 5 true 3 false false true v2-test-z_o_5-pgd_3_no-$TRANSFORM_NAME-$DATASET_NAME_SMALL $TRANSFORM_NAME 0.5 0.5 0.01 $DATASET_NAME 1 $MODEL_NAME $NUM_REPLACE 1
#./experiments/attack_and_test_seq2seq.sh 0 2 1 true 1 5 true 1 false false false v2-5-z_o_5-pgd_1_no-$TRANSFORM_NAME-$DATASET_NAME_SMALL $TRANSFORM_NAME 0.5 0.5 0.01 $DATASET_NAME 1 $MODEL_NAME $NUM_REPLACE 1

#./experiments/attack_and_test_seq2seq.sh 0 2 3 true 1 5 true 3 false false false v2-6-z_o_5-pgd_3_no-$TRANSFORM_NAME-$DATASET_NAME_SMALL $TRANSFORM_NAME 0.5 0.5 0.01 $DATASET_NAME 1 $MODEL_NAME $NUM_REPLACE 1

#./experiments/attack_and_test_seq2seq.sh 0 2 10 true 1 5 true 10 false false false v2-7-z_o_5-pgd_10_no-$TRANSFORM_NAME-$DATASET_NAME_SMALL $TRANSFORM_NAME 0.5 0.5 0.01 $DATASET_NAME 1 $MODEL_NAME $NUM_REPLACE 1

#./experiments/attack_and_test_seq2seq.sh 1 2 20 true 1 5 true 20 false false false v2-8-z_o_5-pgd_20_no-$TRANSFORM_NAME-$DATASET_NAME_SMALL $TRANSFORM_NAME 0.5 0.5 0.01 $DATASET_NAME 1 $MODEL_NAME $NUM_REPLACE 1

# 10 sites

#./experiments/attack_and_test_seq2seq.sh 0 2 3 true 1 10 true 10 false false false v2-9-z_o_10-pgd_3_no-$TRANSFORM_NAME-$DATASET_NAME_SMALL $TRANSFORM_NAME 0.5 0.5 0.01 $DATASET_NAME 1 $MODEL_NAME $NUM_REPLACE 1

# 20 sites

#./experiments/attack_and_test_seq2seq.sh 0 2 3 true 1 20 true 10 false false false v2-91-z_o_20-pgd_3_no-$TRANSFORM_NAME-$DATASET_NAME_SMALL $TRANSFORM_NAME 0.5 0.5 0.01 $DATASET_NAME 1 $MODEL_NAME $NUM_REPLACE 1

# all sites

#./experiments/attack_and_test_seq2seq.sh 0 2 3 true 0 0 true 10 false false false v2-92-z_o_all-pgd_3_no-$TRANSFORM_NAME-$DATASET_NAME_SMALL $TRANSFORM_NAME 0.5 0.5 0.01 $DATASET_NAME 1 $MODEL_NAME $NUM_REPLACE 1

#./experiments/attack_and_test_seq2seq.sh 1 2 20 true 1 1 true 20 false false true v2-96-z_o_1-pgd_20_smooth-$TRANSFORM_NAME-$DATASET_NAME_SMALL $TRANSFORM_NAME 0.5 0.5 0.01 $DATASET_NAME 1 $MODEL_NAME $NUM_REPLACE 1

#./experiments/attack_and_test_seq2seq.sh 1 2 20 true 1 5 true 20 false false true v2-991-z_o_5-pgd_20_smooth-$TRANSFORM_NAME-$DATASET_NAME_SMALL $TRANSFORM_NAME 0.5 0.5 0.01 $DATASET_NAME 1 $MODEL_NAME $NUM_REPLACE 1

#./experiments/attack_and_test_seq2seq.sh 1 2 3 true 1 10 true 3 false false true v2-321-z_o_10-pgd_3_smooth-$TRANSFORM_NAME-$DATASET_NAME_SMALL $TRANSFORM_NAME 0.5 0.5 0.01 $DATASET_NAME 1 $MODEL_NAME $NUM_REPLACE 1

#./experiments/attack_and_test_seq2seq.sh 1 2 3 true 1 5 true 3 false false true v2-3212-z_o_20-pgd_3_smooth-$TRANSFORM_NAME-$DATASET_NAME_SMALL $TRANSFORM_NAME 0.5 0.5 0.01 $DATASET_NAME 1 $MODEL_NAME $NUM_REPLACE 1
