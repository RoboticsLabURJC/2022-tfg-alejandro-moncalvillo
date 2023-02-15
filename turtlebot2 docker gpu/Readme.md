# Docker con Turtlebot2 simulado

Se necesita que esté en una misma carpeta el Dockerfile y la carpeta turtlebot2.
- El docker tiene como base Ubuntu 22.04 e incluye ros2 humble y Gazebo 11

## Modo de uso
Para crear la imagen (base nvidia con ubuntu 22.04)
~~~
./build.sh
~~~

Para crear el contenedor
~~~
sudo docker run --gpus all --rm -it -p 5900:5900 turtlebot2
~~~
o también puedes hacer
~~~
./run.sh
~~~

Para entrar en la máquina con bash:
~~~
$ sudo docker exec -it [ID] /bin/bash
~~~

Para lanzar el robot en gazebo y abrir rviz2
~~~
./empty_world_robot
~~~

Para teleoperar el robot:
~~~
$ ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r cmd_vel:=/cmd_vel -p use_sim_time:=true
~~~
