FROM humble_base

# Turtlebot2 simulated
WORKDIR /home
RUN mkdir -p turtlebot2_ws/src
COPY ./turtlebot2 /home/turtlebot2_ws/src
COPY ./darknet_ros_yolov4/darknet_ros /home/turtlebot2_ws/src/darknet_ros
RUN /home/turtlebot2_ws/src/darknet_ros/rm_darknet_CMakeLists.sh
WORKDIR /home/turtlebot2_ws
RUN sudo rosdep fix-permissions 
RUN rosdep update && rosdep install --from-paths src --ignore-src -r -y

RUN apt update && \
    apt install -y \
        ros-humble-v4l2-camera

RUN colcon build --symlink-install --packages-skip darknet_ros darknet_ros_msgs

COPY empty_world_robot /home/turtlebot2_ws
COPY ./robotics_academy /RoboticsAcademy

RUN pip install websocket
RUN pip install websocket_server
