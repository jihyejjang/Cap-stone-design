import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from utils.video_util import *
import os
import imageio
import glob


def visualize_clip(clip, convert_bgr=False, save_gif=False, file_path=None):
    num_frames = len(clip)
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)

    def update(i):
        if convert_bgr:
            frame = cv2.cvtColor(clip[i], cv2.COLOR_BGR2RGB)
        else:
            frame = clip[i]
        plt.imshow(frame)
        return plt

    # FuncAnimation will call the 'update' function for each frame; here
    # animating over 10 frames, with an interval of 20ms between frames.
    anim = FuncAnimation(fig, update, frames=np.arange(0, num_frames), interval=1)

    if save_gif:
        anim.save(file_path, dpi=80, writer='imagemagick')
    else:
        # plt.show() will just loop the animation forever.
        plt.show()


def visualize_predictions(video_path, predictions, save_path, video_name):
    if not os.path.exists(os.path.join(save_path)):
        save_images_path = os.path.join(save_path, video_name + '_images')
        os.makedirs(save_images_path)

    frames = get_video_frames(video_path)
    assert len(frames) == len(predictions)

    for i in range(0, len(frames), 10):
        fig, ax = plt.subplots(figsize=(5, 5))
        fig.set_tight_layout(True)
        fig_frame = plt.subplot(2, 1, 1)
        fig_prediction = plt.subplot(2, 1, 2)
        fig_prediction.set_xlim(0, len(frames))
        fig_prediction.set_ylim(0, 1.15)

        frame = frames[i]
        x = range(0, i)
        y = predictions[0:i]
        fig_prediction.plot(x, y, '-')
        fig_frame.imshow(frame)
        plt.savefig(save_images_path + r'\{}.png'.format(i))
        plt.cla()

    filenames = glob.glob(save_images_path + r'\*.png')
    filenames.sort(key=lambda x:int(x.split('\\')[-1].split('.')[0]))
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(save_path + r'\{}.gif'.format(video_name), images)

    # ani = FuncAnimation(fig, update, frames=len(frames), interval=1, repeat=False)
    # plt.show()
    # plt.rcParams['animation.ffmpeg_path'] = ffmpeg_path
    # FFwriter = matplotlib.animation.FFMpegWriter(fps=12, extra_args=['-vcodec', 'libx264'])
    # ani.save(save_path, writer=FFwriter)

    # animation.save(save_path)

    # FuncAnimation will call the 'update' function for each frame; here
    # animating over 10 frames, with an interval of 20ms between frames.

    # anim = FuncAnimation(fig, update, frames=np.arange(0, len(frames), 10), interval=1, repeat=False)
    # plt.rcParams['animation.convert_path'] = magick_path
    # writer = matplotlib.animation.ImageMagickFileWriter()
    #
    #     anim.save(save_path, dpi=200, writer='imagemagick')
    #     anim.save(save_path, dpi=200, writer=writer)
    # if save_path:
    #     anim.save(save_path, dpi=200, writer=matplotlib.animation.FFMpegFileWriter())
    # else:
    #     plt.show()
    return


