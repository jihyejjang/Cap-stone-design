import keras
import os
from c3d import *
from classifier import *
from utils.visualization_util import *
import pickle
import glob
import random

def run_demo():
    video_name = os.path.basename(cfg.sample_video_path).split('.')[0]

    # read video
    video_clips, num_frames = get_video_clips(cfg.sample_video_path)

    print("Number of clips in the video : ", len(video_clips))

    if len(video_clips) > 10000:
        return

    # build models
    feature_extractor = c3d_feature_extractor()
    classifier_model = build_classifier_model()

    # print("Models initialized")

    # extract features
    rgb_features = []
    for i, clip in enumerate(video_clips):
        clip = np.array(clip)
        if len(clip) < params.frame_count:
            continue

        clip = preprocess_input(clip)
        rgb_feature = feature_extractor.predict(clip)[0]
        rgb_features.append(rgb_feature)

        # print("Processed clip : ", i)

    rgb_features = np.array(rgb_features)

    # bag features
    rgb_feature_bag = interpolate(rgb_features, params.features_per_bag)

    if cfg.sample_video_path.split('\\')[-2] == 'abnormal':
        with open(r'D:\MK\cctv\vid_features\abnormal\{}.pkl'.format(video_name), 'wb') as f:
            pickle.dump(rgb_feature_bag, f)
    else:
        with open(r'D:\MK\cctv\vid_features\normal\{}.pkl'.format(video_name), 'wb') as f:
            pickle.dump(rgb_feature_bag, f)

    # classify using the trained classifier model
    predictions = classifier_model.predict(rgb_feature_bag)
    
    predictions = np.array(predictions).squeeze()
    
    predictions = extrapolate(predictions, num_frames)
    
    save_path = os.path.join(cfg.output_folder, video_name)
    
    # visualize predictions
    visualize_predictions(cfg.sample_video_path, predictions, save_path, video_name)


if __name__ == '__main__':
    video_list = glob.glob(r'D:\MK\cctv\kindergarden data\abnormal\*.mp4')
    feature_list = glob.glob(r'D:\MK\cctv\vid_features\abnormal\*.pkl')
    feature_list = list(map(lambda x: x.split('\\')[-1].split('pkl')[0], feature_list))
    for video in video_list:
        already = False
        for feature in feature_list:
            if feature in video:
                already = True
        if already ==  True:
            continue
        print(video)
        cfg.sample_video_path = video
        run_demo()

    video_list = glob.glob(r'D:\MK\cctv\kindergarden data\normal\*.mp4')
    feature_list = glob.glob(r'D:\MK\cctv\vid_features\normal\*.pkl')
    feature_list = list(map(lambda x: x.split('\\')[-1].split('pkl')[0], feature_list))
    count = 0
    for video in video_list:
        already = False
        for feature in feature_list:
            if feature in video:
                already = True
        if already ==  True:
            continue
        print(video)
        cfg.sample_video_path = video
        run_demo()
