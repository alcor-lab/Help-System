# deploy
out_H = 112
out_W = out_H
op_input_width = 368
op_input_height = op_input_width
frames_per_step = 6
input_channels = 7
seq_len = 4
enc_fc_1 = 4000
enc_fc_2 = 1500
lstm_units = 800
pre_class = 400
encoder_lstm_layers = 2*[lstm_units]
hidden_states_dim = lstm_units
vocab_len = 34
debug_frames = True
demo_path_video = 'demo_video/'

test_video_path = 'dataset/Video/kit_dataset/TEST_VIDEOS'