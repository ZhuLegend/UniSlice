#!/bin/bash

# set_gpu_percentage.sh

# 函数：获取 MPS 控制器的服务器进程 ID
get_mps_server_pid() {
    echo "get_server_list" | nvidia-cuda-mps-control
}

# 函数：设置 GPU 线程百分比
set_gpu_thread_percentage() {
    local percentage=$1
    local server_pid=$(get_mps_server_pid)

    if [ -z "$server_pid" ]; then
        echo "MPS server PID not found."
        return 1
    fi

    echo "set_active_thread_percentage $server_pid $percentage" | nvidia-cuda-mps-control
}

# 主程序
# 检查是否传入了百分比参数
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <percentage>"
    exit 1
fi

percentage=$1

# 调用函数设置 GPU 线程百分比
set_gpu_thread_percentage "$percentage"