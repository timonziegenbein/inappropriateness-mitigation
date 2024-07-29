for i in {0..0}
do
    for j in {0..4}
    do
        export CUDA_VISIBLE_DEVICES=0
        python /bigwork/nhwpziet/appropriateness-style-transfer/src/annotation-study-appropriateness-prediction/binary_debertav3.py --fold ${j} --repeat ${i} --output /bigwork/nhwpziet/appropriateness-style-transfer/data/models/binary-debertav3-conservative-no-issue --input /bigwork/nhwpziet/arg-appropriateness-final/data/appropriateness-corpus/appropriateness_corpus_conservative_w_folds.csv --issue
    done
done

