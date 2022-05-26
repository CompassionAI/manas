BASEDIR=$1
MAX_SEQ_LEN=$2
MODEL_PREFIX=$3

proc_path () {
    local path=$1
    file_name=${path##*/}
    new_name="$(cut -d'.' -f1 <<<"$file_name")"-maxseq-${MAX_SEQ_LEN}.tf_record
    if [ -f "${BASEDIR}/${new_name}" ]; then
        echo "${BASEDIR}/${new_name} exists."
    else
        # echo ${path}
        # echo ${file_name}
        echo from ${BASEDIR}/${file_name}  --  to ${BASEDIR}/${new_name}

        python3 -m albert.create_pretraining_data \
            --input_file=${BASEDIR}/${file_name} \
            --output_file=${BASEDIR}/${new_name} \
            --spm_model_file=./sp_models/${MODEL_PREFIX}.model \
            --vocab_file=./sp_models/${MODEL_PREFIX}.vocab \
            --do_lower_case=False \
            --keep_accents=True \
            --stochastic_tokens=True \
            --max_seq_length=${MAX_SEQ_LEN}\
            --max_predictions_per_seq=20 \
            --masked_lm_prob=0.15 \
            --random_seed=12345 \
            --offset_steps=10 \
            --offset_stride=3 \
            --dupe_factor=4 \
            --do_whole_word_mask=False \
            --favor_shorter_ngram=True \
            --do_permutation=False \
            --random_next_sentence=False \
            --shuffle_examples=True
    fi
}

for path in $BASEDIR/*.txt
    do
        proc_path "$path" &
    done
