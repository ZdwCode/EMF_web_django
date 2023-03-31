from os import path, chdir
import sys

# 将manage.py所在的目录添加到系统路径
sys.path.append(path.dirname(path.abspath(__file__)))

# 将工作目录更改为manage.py所在的目录
chdir(path.dirname(path.abspath(__file__)))

from django.core.management import execute_from_command_line

if __name__ == '__main__':
    execute_from_command_line(sys.argv)
