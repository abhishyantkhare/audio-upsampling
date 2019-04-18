export PYTHONPATH="$PYTHONPATH:~/Documents/Projects/Personal/audio-upsampling/model/models-master/"
python3 official/transformer/upsample.py --file ../../datasets/eval/jlucas_EVAL.tfrecord --file_out jlucas_upsample.txt  
