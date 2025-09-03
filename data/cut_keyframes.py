import argparse
import json
import os
import numpy as np
import imageio

def cut_keyframes(video_dir, qa_id, start_frame_id, end_frame_id, frame_number, keyframes_dir):
    frame_idx = np.linspace(start_frame_id, end_frame_id, frame_number, endpoint=True, dtype=int)
    print(f"start frame id: {start_frame_id}, end frame id: {end_frame_id}, sampled frames: {frame_idx}")

    save_dir = os.path.join(keyframes_dir, qa_id)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    video_path = os.path.join(video_dir, qa_id +'.mp4')
    try:
        clip = imageio.get_reader(video_path)

        for idx, frame_id in enumerate(frame_idx):
            frame = clip.get_data(frame_id)
            image_path = os.path.join(save_dir, f'frame-{idx}_frameID-{frame_id}.png')
            imageio.imwrite(image_path, frame)
            print('----------------------------------------')
            print(f"Saved: {image_path}")
    except Exception as e:
        print(f"Error processing video {video_path}: {e}")

    
def get_current_obs(video_dir, qa_id, keyframes_dir):
    save_dir = os.path.join(keyframes_dir, qa_id)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    video_path = os.path.join(video_dir, qa_id +'.mp4')
    try:
        clip = imageio.get_reader(video_path)
        frame = clip.get_data(0)
        image_path = os.path.join(save_dir, f'frameID-{0}.png')
        imageio.imwrite(image_path, frame)
        print('----------------------------------------')
        print(f"Saved: {image_path}")
    except Exception as e:
        print(f"Error processing video {video_path}: {e}")

def main(qa_anno, video_dir, keyframes_dir, frame_number):
    for qa_item in qa_anno:
        qa_id = qa_item['sample_id']
        if not os.path.exists(os.path.join(keyframes_dir, qa_id)):
            os.makedirs(os.path.join(keyframes_dir, qa_id))
        if len(os.listdir(os.path.join(keyframes_dir, qa_id))) == 0:  # 空文件夹
            get_current_obs(video_dir, qa_id, keyframes_dir)

# 示例调用
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Arg Parser')
    parser.add_argument('--video_dir', type=str, default='~/Documents/data/BAAI_nav/L_Video2/videos')
    parser.add_argument('--anno_path', type=str, default='data/BAAI-navi/BAAI-navi.json')
    parser.add_argument('--visual_input_dir', type=str, default='visual')
    parser.add_argument('--frame_number', type=int, default=8) # only for image models
    args = parser.parse_args()

    qa_anno = json.load(open(args.anno_path))
    video_dir = args.video_dir
    keyframes_dir = os.path.join(args.visual_input_dir, 'keyframes_dir')
    frame_number = args.frame_number

    print(f"keyframes_dir: {keyframes_dir}")
    if not os.path.exists(keyframes_dir):
        print(f"Create directory: {keyframes_dir}")
        os.makedirs(keyframes_dir)
    
    main(qa_anno, video_dir, keyframes_dir, frame_number)