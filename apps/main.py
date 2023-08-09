# utf-8
import os
import sys
import traceback

import setLog
import displayGUI
import variables
import fileClassification
import PySimpleGUI as SG
from CONST import *

info_log = logging.getLogger(INFO_LOG)
info_log.setLevel(logging.INFO)
error_log = logging.getLogger(ERROR_LOG)
error_log.setLevel(logging.ERROR)


if __name__ == '__main__':
	variables.ab_path = os.getcwd()
	args = sys.argv
	setLog.set_log()
	info_log.info("\n---START PROGRAM---")
	if len(args) > 1:
		if args[1] == 'test':
			info_log.info("~~~テストモード~~~~")
			variables.test_flag = True
	try:
		frame = displayGUI.MainGUI()
		file = frame.main_loop()
		fileClassification.divide_files(file)
	except Exception as e:
		SG.popup_error(f"{ERROR_MESS[1]}\n\n{repr(e)}", title=ERROR_MESS[0])
		info_log.info("---ERROR OCCURRED---")
		sys.exit(error_log.exception(f"致命的なエラー\n{traceback.format_exc()}"))
	else:
		os.startfile(variables.OUTPUT_PATH)
		SG.popup_ok(COMP_MESS[1], title=COMP_MESS[0], keep_on_top=True)
		info_log.info("---FINISH PROGRAM---")
	finally:
		info_log.info("\n---CLOSE PROGRAM---")

