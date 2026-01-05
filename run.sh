#!/bin/bash

# 配置
MODEL_PATH="models/whisper-large-v3-turbo-finetune"
HOST="0.0.0.0"
PORT=5000
PID_FILE="whisper.pid"
LOG_DIR="logs"

# 创建日志目录（如果不存在）
mkdir -p $LOG_DIR

# 获取当天日志文件名
get_log_file() {
    echo "$LOG_DIR/whisper-$(date +%Y-%m-%d).log"
}

start() {
    if [ -f "$PID_FILE" ] && kill -0 $(cat "$PID_FILE") 2>/dev/null; then
        echo "服务已经在运行，PID=$(cat $PID_FILE)"
        exit 1
    fi

    LOG_FILE=$(get_log_file)
    echo "启动服务... 日志: $LOG_FILE"
    nohup python3 infer_server.py --host=$HOST --port=$PORT --model_path=$MODEL_PATH >> "$LOG_FILE" 2>&1 &
    echo $! > $PID_FILE
    echo "服务启动完成，PID=$(cat $PID_FILE)"
}

stop() {
    if [ ! -f "$PID_FILE" ]; then
        echo "PID 文件不存在，服务未启动？"
        exit 1
    fi

    PID=$(cat "$PID_FILE")
    if kill -0 $PID 2>/dev/null; then
        echo "停止服务，PID=$PID ..."
        kill $PID
        rm -f $PID_FILE
        echo "服务已停止"
    else
        echo "服务进程不存在，清理 PID 文件"
        rm -f $PID_FILE
    fi
}

status() {
    if [ -f "$PID_FILE" ] && kill -0 $(cat "$PID_FILE") 2>/dev/null; then
        echo "服务正在运行，PID=$(cat $PID_FILE)"
    else
        echo "服务未运行"
    fi
}

restart() {
    stop
    sleep 1
    start
}

case "$1" in
    start)
        start
        ;;
    stop)
        stop
        ;;
    restart)
        restart
        ;;
    status)
        status
        ;;
    *)
        echo "用法: $0 {start|stop|restart|status}"
        exit 1
        ;;
esac
