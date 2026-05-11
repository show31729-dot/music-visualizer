#!/bin/bash
# 雙擊這個檔案就可以啟動 Music Visualizer

# 切換到腳本所在資料夾
cd "$(dirname "$0")"

echo "🎵 Music Visualizer 啟動中..."
echo ""

# 檢查 Python3
if ! command -v python3 &>/dev/null; then
  echo "❌ 找不到 Python3，請先安裝："
  echo "   https://www.python.org/downloads/"
  read -p "按 Enter 關閉..."
  exit 1
fi

# 自動安裝缺少的套件
echo "📦 檢查套件..."
python3 -c "import flask, librosa, cv2, moviepy, tqdm" 2>/dev/null
if [ $? -ne 0 ]; then
  echo "⚙️  首次使用，安裝必要套件（約需 1~2 分鐘）..."
  pip3 install -r requirements.txt -q
  echo "✅ 安裝完成！"
fi

# 確保資料夾存在
mkdir -p uploads outputs

# 關掉舊的伺服器（如果有）
lsof -ti:5001 | xargs kill -9 2>/dev/null

# 啟動伺服器（背景執行）
echo ""
echo "🚀 啟動伺服器..."
python3 app.py &
SERVER_PID=$!

# 等待伺服器就緒
sleep 2

# 自動開瀏覽器
echo "🌐 開啟瀏覽器..."
open http://127.0.0.1:5001

echo ""
echo "✅ Music Visualizer 已啟動！"
echo "   瀏覽器應該已自動開啟 http://127.0.0.1:5001"
echo ""
echo "⚠️  關閉此視窗會停止伺服器"
echo "   （使用完畢後可以直接關閉這個視窗）"
echo ""

# 等待伺服器（按 Ctrl+C 或關閉視窗時自動停止）
wait $SERVER_PID
