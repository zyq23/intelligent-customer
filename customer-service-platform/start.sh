#!/bin/bash

# ============================================================
# 问道星辰智能客服系统 - 一键启动脚本
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "  问道星辰智能客服系统 - 快速启动"
echo "============================================================"
echo ""

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 检查 .env 文件
if [ ! -f "backend/.env" ]; then
    echo -e "${RED}错误：backend/.env 文件不存在${NC}"
    echo "请从 backend/.env.example 复制并配置:"
    echo "  cp backend/.env.example backend/.env"
    exit 1
fi

# 检查必要的环境变量
check_env_var() {
    if [ -z "$1" ]; then
        echo -e "${YELLOW}警告：$2 未配置${NC}"
        return 1
    fi
    return 0
}

echo -e "${GREEN}[1/5] 检查配置文件...${NC}"
DEEPSEEK_API_KEY=$(grep "DEEPSEEK_API_KEY=" backend/.env | cut -d'=' -f2)
DB_PASSWORD=$(grep "DB_PASSWORD=" backend/.env | cut -d'=' -f2)
SECRET_KEY=$(grep "SECRET_KEY=" backend/.env | cut -d'=' -f2)

check_env_var "$DEEPSEEK_API_KEY" "DEEPSEEK_API_KEY"
check_env_var "$DB_PASSWORD" "DB_PASSWORD"

if [ "$SECRET_KEY" = "your_secret_key_change_in_production" ] || [ "$SECRET_KEY" = "your-secret-key" ]; then
    echo -e "${RED}⚠️  安全警告：SECRET_KEY 使用默认值，请立即修改！${NC}"
fi

# 创建必要的目录
echo -e "${GREEN}[2/5] 创建必要的目录...${NC}"
mkdir -p uploads
mkdir -p backend/app/graphrag/data/input
mkdir -p backend/app/graphrag/data/output
mkdir -p logs

# 检查 Docker 依赖
echo -e "${GREEN}[3/5] 检查服务依赖...${NC}"
check_port() {
    if netstat -tuln | grep -q ":$1 "; then
        echo "  ✓ 端口 $1 已被使用"
        return 0
    else
        echo -e "  ${YELLOW}✗ 端口 $1 未被使用${NC}"
        return 1
    fi
}

# MySQL (3307)
check_port 3307 || echo -e "  ${YELLOW}提示：启动 MySQL: docker compose -f deploy/docker-compose.yml up -d mysql${NC}"

# Redis (6381)
check_port 6381 || echo -e "  ${YELLOW}提示：启动 Redis: docker compose -f deploy/docker-compose.yml up -d redis${NC}"

# Neo4j (7687)
check_port 7687 || echo -e "  ${YELLOW}提示：启动 Neo4j: docker compose -f deploy/docker-compose.yml up -d neo4j${NC}"

# Ollama (11434)
check_port 11434 || echo -e "  ${YELLOW}提示：如需本地模型，请安装 Ollama${NC}"

# 激活虚拟环境
echo -e "${GREEN}[4/5] 激活 Python 虚拟环境...${NC}"
if [ -d "backend/.venv" ]; then
    source backend/.venv/bin/activate
    echo "  ✓ 虚拟环境已激活"
else
    echo -e "${RED}✗ 虚拟环境不存在${NC}"
    echo "  请先运行：cd backend && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# 启动后端服务
echo -e "${GREEN}[5/5] 启动后端服务...${NC}"
echo "  访问地址：http://localhost:9003"
echo "  API 文档：http://localhost:9003/docs"
echo ""
echo -e "${YELLOW}按 Ctrl+C 停止服务${NC}"
echo "============================================================"

cd backend
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 9003 --reload
