CUDA_VISIBLE_DEVICES="0" python run_lstm.py --sequence_length 1 &
CUDA_VISIBLE_DEVICES="0" python run_lstm.py --sequence_length 3 &
CUDA_VISIBLE_DEVICES="0" python run_lstm.py --sequence_length 5 &
CUDA_VISIBLE_DEVICES="1" python run_lstm.py --sequence_length 10 &
CUDA_VISIBLE_DEVICES="1" python run_lstm.py --sequence_length 20 &
CUDA_VISIBLE_DEVICES="1" python run_lstm.py --sequence_length 30 &
