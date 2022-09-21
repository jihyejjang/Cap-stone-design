import os

root = os.getcwd()

c3d_model_weights = root + r'\trained_models\c3d_sports1m.h5'
classifier_model_weigts = root + r'\trained_models\weights_L1L2'
classifier_model_json = root + r'\trained_models\model.json'

input_folder  = root + r'\input'
output_folder = root + r'\output'

sample_video_path = root + r'\input\Explosion008_x264.mp4'

ffmpeg_path = root + r'\ffmpeg-4.2.2-win64-static\bin\ffmpeg.exe'