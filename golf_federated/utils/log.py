# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2022/6/1 16:39
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2022/6/16 0:55

import sys
from loguru import logger
import datetime


class LoggerHere(object):
    """

    A log class customized for GOLF.

    """

    def __init__(self) -> None:
        """

        Initialize class object.

        """

        # Get time for saving logs.
        now = datetime.datetime.now()

        # Get logger object in loguru library.
        self.logger = logger

        # Clear the settings of the logger object.
        self.logger.remove()

        # Add log output to command box.
        self.logger.add(
            sink=sys.stderr,
            format="  <level>{level.icon}  {level}</level> "
                   "<w>|</w> "
                   "<cyan>{time:YYYYMMDD HH:mm:ss}</cyan> "
                   "<w>|</w> "  # 颜色>时间
                   "<cyan>{process.name}</cyan> "
                   "<w>|</w> "  # 进程名
                   "<cyan>{thread.name}</cyan> "
                   "<w>|</w> "  # 进程名
                   "<green>{module}</green>"
                   "<w>.</w>"
                   "<green>{function}</green>"  # 模块名.方法名
                   "<w>:</w>"
                   "<green>{line}</green> "
                   "<w>|</w> "  # 行号
                   "<level>{message}</level>",  # 日志内容
        )

        # Add log output to specified file.
        self.logger.add(
            sink="./testlog/test" + now.strftime("%Y_%m_%d") + ".log",
            format="  {level.icon}  {level} "
                   "| "
                   "{time:YYYYMMDD HH:mm:ss} "
                   "| "  # 颜色>时间
                   "{process.name} "
                   "| "  # 进程名
                   "{thread.name} "
                   "<w>|</w> "  # 进程名
                   "{module}"
                   "."
                   "{function}"  # 模块名.方法名
                   ":"
                   "{line} "
                   "| "  # 行号
                   "{message}",  # 日志内容
            encoding='utf-8',
            rotation="500MB"
        )

        # Add different log input levels.
        self.logger.level("Client Info  ", no=10, color="<e><b>", icon="📱")
        self.logger.level("Server Info  ", no=11, color="<m><b>", icon="💻")
        self.logger.level("Please Note  ", no=12, color="<y><b>", icon="⚠")
        self.logger.level("Common Info  ", no=13, color="<b>", icon="⚪")
        self.logger.level("Error Message", no=13, color="<r><v>", icon="❌")


# Execute log object.
loggerhear = LoggerHere().logger
