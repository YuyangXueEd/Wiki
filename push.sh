./cp.sh 

# 复制 README.md
# cp docs/README.md README.md

# 更新 master
git add .
git commit -m "${1}"
git push