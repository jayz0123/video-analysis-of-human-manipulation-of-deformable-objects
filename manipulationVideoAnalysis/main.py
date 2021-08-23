from utils.video_handler import VideoHandler

if __name__ == '__main__':
    videos = VideoHandler(
        rgb_path='resources/test_frames/rgb/',
        depth_path='resources/test_frames/depth/',
        hole_detecting=True,
        hand_detecting=True,
        aglet_detecting=True)

    videos.replay('rgb')
