from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
from utils import TeamAssigner
from utils import PlayerBallAssigner
from utils import CameraMovementEstimator
from view_transformer import ViewTransformer
from utils import SpeedAndDistance_Estimator
from huggingface_hub import hf_hub_download


def main():
    # Read Video
    # video_frames = read_video('input_videos/08fd33_4.mp4')
    video_frames = read_video('input_videos/game0.mp4')

    #download already finetuned model
    model=hf_hub_download('abhi1304/Automated-Football-Analysis','YOLOv5_pretrained_on_football.pt')

    # Initialize Tracker
    tracker = Tracker(model)

    tracks = tracker.get_object_tracks(video_frames,              #simply returns the position coordinates of players, referee and ball 
                                    read_from_stub=False,                 #for each frame in format track["player"][frame_n][track_id]
                                    stub_path='stubs/track_stubs.pkl')
                                    
    # Get object positions 
    tracker.add_position_to_tracks(tracks)

    # camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                                read_from_stub=True,
                                                                                stub_path='stubs/camera_movement_stub.pkl')
    #has camera position for each frame relative to first frame of video
    
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame)
    #players position in tracks now contain position relative to camera movement


    # View Trasnsformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Speed and distance estimator
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], 
                                    tracks['players'][0])
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],   
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    
    # Assign Ball Aquisition
    player_assigner =PlayerBallAssigner()
    team_ball_control= []      #Keep track of which team has ball in the frame.
    
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:   #In case ball is far from all players in a frame so not assigned to any player.
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            if len(team_ball_control)==0 :
                team_ball_control.append(1)
            else:
                team_ball_control.append(team_ball_control[-1])
    team_ball_control= np.array(team_ball_control)


    # Draw output 
    ## Draw object Tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks,team_ball_control)

    ## Draw Camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames,camera_movement_per_frame)

    ## Draw Speed and Distance
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames,tracks)

    # Save video
    save_video(output_video_frames, 'output_videos/output0.avi')

if __name__ == '__main__':
    main()
