This is a pytorch real time object detection implementation using pc camera.
I personally used MaskRcnn provied by pytorch and its surroundings such as coco label.

It's highly recommended that you run this project with cuda, otherwise real-time detection becomes real-slow-time detection.

There are two versions of object detection, one is one-picture detection, in one_picture_version.py, the other is real-time camera object detection, in realtime_detect.py. Both of them are able to run independently.

