# utf-8
import os
import sys

from CONST import *


def set_log() -> None:
	"""
	ログの設定
	* 標準ログとエラーログの2種類を設定している
	:return: None
	"""
	if not os.path.exists(LOG_PATH):  # ログフォルダがなかったら作成
		os.mkdir(LOG_PATH)
	info_handler = logging.FileHandler(filename=LOG_FILENAME % 'STAND', encoding=UTF8, mode='a')
	info_handler.setLevel(logging.INFO)
	info_handler.setFormatter(logging.Formatter(LOG_FORMAT))
	info_handler.addFilter(lambda record: record.levelno <= logging.INFO)
	error_handler = logging.FileHandler(filename=LOG_FILENAME % 'ERROR', encoding=UTF8, mode='a')
	error_handler.setLevel(logging.ERROR)
	error_handler.setFormatter(logging.Formatter(LOG_FORMAT))
	stream_handler = logging.StreamHandler(sys.stderr)
	stream_handler.setFormatter(logging.Formatter(LOG_FORMAT))
	stream_handler.setLevel(logging.ERROR)
	info_log.addHandler(info_handler)
	error_log.addHandler(error_handler)
	error_log.addHandler(stream_handler)