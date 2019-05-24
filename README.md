# Help-System

## Installation
1) downloade the repository 
2) create "\model" folder in the repository
4) download the network files from this link and place it in the "\model" folder.
https://drive.google.com/drive/folders/1aSM6h17ZQYegrOA8IK_qw2eN_nF4rOWV?usp=sharing

## Usage and functions
Here is a list of the usefull functions contained in the class. to use start by calling the class:

```python
import activity_network
  
helpnet = activity_network.activity_network()
```

```python
flow = helpnet.compute_optical_flow(frame, frame_prev)
```

returns the optical flow given two frame
  
```python
pafMat, heatMat = helpnet.compute_pose(img)
```
  
returns openPose pafMat and heatMat
    
```python
frame_preprocessed = helpnet.compound_channel(img, flow, heatMat, pafMat)
```
 
stack the rgb image, the optical flow, the heatMat and the pafMat along the channel dimension returning a 7 channel picture.

```python
second_matrix = helpnet.compound_second_frames(frames)
```

groups into correct format a list "frames" of preprocessed frame.
Temporaly ordered such that the first frame in the list is the 
oldest while the last is the newest.

```python
input_tensor = helpnet.create_input_tensor_given_seconds(seconds)
```

create network input tensor given list of correctly formatted 
second matrices. Temporaly ordered such that the the first matrix
in the list must be the oldest, the last the newest.

```python
input_tensor = helpnet.create_input_tensor_given_preprocessed_frame(frames_collection)
```
create network input tensor given list of list of preprocessed frames.
frames_collection must be a list of list of frames, where each list of frame
contains the list of preprocessed frame extracted from one second. the frames
must be ordered such that the first frame must be the oldest and the last the
newest. The list of those list must follow the same order with the first list
containing the frame of the oldest second and the last the one of the newest

```python
now_softmax, next_softmax, help_softmax = helpnet.compute_activity_given_tensor(input_tensor, second_count)
```

returns the softmax for the now and next activity and the help action. It requires in iput the input tensor and the second count of the newest given second.

```python
now_softmax, next_softmax, help_softmax = helpnet.compute_activity_given_seconds_matrix(input_tensor, second_count)
```
return the outputs given the list of seconds matrix

```python
now_softmax, next_softmax, help_softmax = helpnet.compute_activity_given_frame_list(input_tensor, second_count)
```

return the outputs given the list of frames matrix

