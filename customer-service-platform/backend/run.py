"""服务启动入口

用法:
    python run.py                 # 开发模式(热重载, 端口 9002)
    python run.py --port 9002     # 指定端口
"""
import os
from pathlib import Path

import uvicorn

from app.core.logger import get_logger

logger = get_logger(service="server")


def start_server(host: str = "0.0.0.0", port: int = 9003, reload: bool = True):
    """启动服务"""
    # 确保工作目录正确(便于相对路径如 logs/uploads 的解析)
    os.chdir(Path(__file__).parent)

    logger.info("Starting intelligent-customer-service...")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Service will listen on http://{host}:{port}")

    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        access_log=False,
        log_level="info",
        reload=reload,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="问道星辰智能客服服务")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=9003, help="监听端口")
    parser.add_argument("--no-reload", action="store_true", help="关闭热重载(生产模式)")
    args = parser.parse_args()

    start_server(host=args.host, port=args.port, reload=not args.no_reload)
