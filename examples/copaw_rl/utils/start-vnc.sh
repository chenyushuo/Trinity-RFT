#!/bin/bash

# 启动虚拟显示
Xvfb :1 -screen 0 1280x800x24 &
export DISPLAY=:1

# 等待 Xvfb 启动
sleep 2

# 启动桌面环境
startxfce4 &

# 启动 VNC 服务器
x11vnc -display :1 -forever -nopw -rfbport 5900 &

# 启动 noVNC（Web 访问）
websockify --web /usr/share/novnc/ 6080 localhost:5900 &

# 保持运行
wait
