# Help-System

Usefull functions:

  compute_optical_flow(frame, frame_prev)
    returns the optical flow given two frame
  
  compute_pose(img)
    returns openPose pafMat and heatMat
    
  compound_channel(img, flow, heatMat, pafMat)
    stack the rgb image, the optical flow, the heatMat and the pafMat along the channel dimension returning a
    7 channel picture.
    
  compound_second_frames(frames)
    groups into correct format a list "frames" of preprocessed frame.
    Temporaly ordered such that the first frame in the list is the 
    oldest while the last is the newest.
    
  create_input_tensor_given_seconds(seconds)
    create network input tensor given list of correctly formatted 
    second matrices. Temporaly ordered such that the the first matrix
    in the list must be the oldest, the last the newest.
    
  create_input_tensor_given_preprocessed_frame(frames_collection)
    create network input tensor given list of list of preprocessed frames.
    frames_collection must be a list of list of frames, where each list of frame
    contains the list of preprocessed frame extracted from one second. the frames
    must be ordered such that the first frame must be the oldest and the last the
    newest. The list of those list must follow the same order with the first list
    containing the frame of the oldest second and the last the one of the newest
