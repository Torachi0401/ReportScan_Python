import sys

import PySimpleGUI as sg

from CONST import *

info_log = logging.getLogger(INFO_LOG)
info_log.setLevel(logging.INFO)
error_log = logging.getLogger(ERROR_LOG).getChild(__name__)
error_log.setLevel(logging.ERROR)
# 呼び出し元でハンドラが設定されない限りログを出力しない
logging.getLogger(__package__).addHandler(logging.NullHandler())


def rewrite_ini(_which: str = None, _changed_value: str = None):
	"""
	Config.iniを書き換える処理
	:param _which:
	:param _changed_value:
	:return:
	"""
	CONFIG_INI.read(CONFIG_PATH, UTF8)
	if _which == "Display":
		print(_which)
		print(_changed_value)
		CONFIG_INI[_which]["show"] = _changed_value
	elif "Path" in _which:
		CONFIG_INI[_which]["Path"] = _changed_value
	try:
		with open(CONFIG_PATH, mode='w', encoding=UTF8) as configfile:
			# 指定したconfigファイルを書き込み
			CONFIG_INI.write(configfile)
		CONFIG_INI.read(CONFIG_PATH, UTF8)
	except Exception as err:
		sg.popup_error(f"{UNREWRITED_MESS}\n\n{repr(err)}")
		sys.exit(error_log.exception(UNREWRITED_MESS))


def reset_default():
	CONFIG_INI.read(CONFIG_PATH, UTF8)
	try:
		for i, v in DEFAULT_SETTING.items():
			CONFIG_INI[SETTINGS][i] = str(v)
		with open(CONFIG_PATH, mode='w', encoding=UTF8) as configfile:
			CONFIG_INI.write(configfile)
		info_log.info("設定内容を初期値に戻す")
		sys.exit(sg.popup(RESTART_MES[1], title=RESTART_MES[0], keep_on_top=True))
	except Exception as err:
		sg.popup_error(f"{UNREWRITED_MESS}\n\n{repr(err)}")
		sys.exit(error_log.exception(UNREWRITED_MESS))


def change_settings(_dict):
	CONFIG_INI.read(CONFIG_PATH, UTF8)
	try:
		for i, v in _dict.items():
			CONFIG_INI[SETTINGS][i] = v
		with open(CONFIG_PATH, mode='w', encoding=UTF8) as configfile:
			CONFIG_INI.write(configfile)
		info_log.info(f"読取設定の変更\n変更箇所{_dict}")
		sys.exit(sg.popup(RESTART_MES[1], title=RESTART_MES[0], keep_on_top=True))
	except Exception as err:
		sg.popup_error(f"{UNREWRITED_MESS}\n\n{repr(err)}")
		sys.exit(error_log.exception(UNREWRITED_MESS))

