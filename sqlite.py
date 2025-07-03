import sqlite3

# 创建（或打开）一个 SQLite 数据库文件
conn = sqlite3.connect('my_database.sqlite')

# 关闭连接
conn.close()
