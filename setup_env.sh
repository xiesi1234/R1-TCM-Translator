# Python dependencies
pip install -r requirements.txt

# Optional system font package for rendering CJK text in some environments
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y fonts-noto-cjk
else
  echo "apt-get not found. Skip installing fonts-noto-cjk."
fi

echo "GRPO environment setup complete."
