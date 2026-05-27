#!/bin/bash

echo "========================================"
echo "       Git 自动化上传脚本"
echo "========================================"
echo ""

DEFAULT_BRANCH="master"
BRANCH=${1:-$DEFAULT_BRANCH}

echo "[INFO] 目标分支: $BRANCH"
echo "[INFO] 检查远程仓库连接状态..."
echo ""

REMOTE_URL=$(git remote get-url origin 2>/dev/null)
if [ -z "$REMOTE_URL" ]; then
    echo "[ERROR] 未检测到远程仓库关联，请先关联远程仓库"
    exit 1
fi
echo "[INFO] 已关联远程仓库: $REMOTE_URL"
echo ""

echo "[STEP 1] 执行 git add . 命令..."
git add .
if [ $? -ne 0 ]; then
    echo "[ERROR] git add 操作失败，请检查文件权限或仓库状态"
    exit 1
fi
echo "[SUCCESS] 文件暂存完成"
echo ""

echo "请输入commit备注信息："
read COMMIT_MSG

while [ -z "$COMMIT_MSG" ]; do
    echo "[ERROR] commit备注信息不能为空，请重新输入："
    read COMMIT_MSG
done

echo ""
echo "[STEP 2] 执行 git commit 命令..."
git commit -m "$COMMIT_MSG"
if [ $? -ne 0 ]; then
    echo "[ERROR] git commit 操作失败，请检查暂存区是否有变更"
    exit 1
fi
echo "[SUCCESS] 提交完成"
echo ""

echo "[STEP 3] 执行 git push 命令..."
git push origin $BRANCH
if [ $? -ne 0 ]; then
    echo "[ERROR] git push 操作失败，请检查网络连接或仓库权限"
    exit 1
fi
echo "[SUCCESS] 代码已成功推送到 $BRANCH 分支"
echo ""
echo "========================================"
echo "       上传完成！"
echo "========================================"