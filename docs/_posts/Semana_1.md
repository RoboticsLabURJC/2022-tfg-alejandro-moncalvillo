---
title: "Semana 1: Iniciarse con ROS2 Humble, pruebas de turtlebot2 y Tello"
categories:
  - Weblog
tags:
  - ROS2 Humble
  - Gazebo
  - Turtlebot2
  - Tello
---

En primer lugar he instalado la distribución Humble de ROS2. Después a partir del repositorio del  ([TFG de Carlos Caminero](https://github.com/RoboticsLabURJC/2021-tfg-carlos-caminero)). Se ha conseguido simular el turtlebot2 en Gazebo 11.10.

Un par de días después hemos probado los drivers provistos por ([IntelligentRoboticsLab](https://github.com/IntelligentRoboticsLabs/Robots/tree/humble)), en el robot físico. Aparentemente parecen funcionar de manera correcta tanto los de la base kobuki como los del láser.

Por último he conseguido compilar el paquete de ([ROS2 tello_driver](http://wiki.ros.org/tello_driver)) en la distribución Humble. El siguiente paso, al igual que con el turtlebot2, es probarlo en el robot físico.
