#!/bin/bash
# GraphRAG 模型下载和索引构建监控脚本

LOG_FILE="/tmp/bge_m3_download.log"
CHECK_INTERVAL=60  # 检查间隔（秒）

echo "==================================="
echo "GraphRAG Embedding 模型下载监控"
echo "==================================="
echo ""
echo "模型：bge-m3 (~1.2GB)"
echo "日志：$LOG_FILE"
echo "检查间隔：${CHECK_INTERVAL}s"
echo ""

# 显示实时进度
while true; do
    clear
    echo "=== $(date '+%Y-%m-%d %H:%M:%S') ==="
    echo ""
    
    if [ ! -f "$LOG_FILE" ]; then
        echo "日志文件不存在，下载可能还未开始..."
        sleep $CHECK_INTERVAL
        continue
    fi
    
    # 读取最新进度行
    PROGRESS=$(tail -5 "$LOG_FILE" | grep -E "(pulling|complete|success)" | tail -1)
    
    if [ -n "$PROGRESS" ]; then
        echo "当前进度:"
        echo "$PROGRESS"
        echo ""
    fi
    
    # 检查是否完成
    if grep -q "total download time" "$LOG_FILE"; then
        echo "✅ 模型下载完成！"
        echo ""
        echo "接下来将自动执行："
        echo "1. 验证模型可用性"
        echo "2. 更新配置文件为 Ollama embedding"
        echo "3. 重新构建 GraphRAG 索引"
        break
    fi
    
    # 显示下载信息
    DOWNLOAD_INFO=$(tail -20 "$LOG_FILE" | grep -oP '\d+%\s+\S+' | tail -1)
    SPEED_INFO=$(tail -20 "$LOG_FILE" | grep -oP '[\d.]+\s*(MB/s|KB/s)' | tail -1)
    
    if [ -n "$DOWNLOAD_INFO" ] && [ -n "$SPEED_INFO" ]; then
        echo "下载速度：$SPEED_INFO"
        echo "预计剩余时间：$(tail -20 "$LOG_FILE" | grep -oP '\d+m\d+s' | tail -1)"
    fi
    
    echo ""
    echo "按 Ctrl+C 终止下载，或等待自动完成..."
    sleep $CHECK_INTERVAL
done

echo ""
echo "开始重建索引..."
# 可以在这里添加自动重建索引的命令
