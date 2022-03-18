rm -rf docs/
cp -R /Users/rasin/Library/Mobile\ Documents/com~apple~CloudDocs/RsKnowledgeBase/Rs docs/
rm -rf docs/.obsidian
rm -rf docs/.trash
rm -rf .DS_Store

find "./docs" -name .DS_Store | xargs rm -rf
