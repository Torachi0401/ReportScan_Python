# utf-8
import itertools
import math
import os
import shutil

from CONST import *
import variables

info_log = logging.getLogger(INFO_LOG)
info_log.setLevel(logging.INFO)
error_log = logging.getLogger(ERROR_LOG).getChild(__name__)
error_log.setLevel(logging.ERROR)
# 呼び出し元でハンドラが設定されない限りログを出力しない
logging.getLogger(__package__).addHandler(logging.NullHandler())


def divide_files(rename_files):
	"""
	読取結果に基づき、ファイルをリネームしてフォルダに分ける（無ければ作成する）
	:return: None
	"""
	for old_path, new in rename_files:
		datetimes = datetime.today().strftime('%Y%m%d%H%M%S')
		if isinstance(new, float):  # エラーの画像はnanでfloat型のため、ここで判定している
			if math.isnan(new):
				pass  # エラーの場合は、元のフォルダから移動させないので、passしている
		else:
			new_path = os.path.join(variables.READ_PATH, str(new)[2:-2] + "_" + datetimes + EXT_PDF)
			if os.path.exists(new_path):
				dirpath, filename = os.path.split(new_path)
				name, ext = os.path.splitext(filename)
				for i in itertools.count(1):
					newname = '{}_{}{}'.format(name, i, ext)
					target_name = os.path.join(dirpath, newname)
					os.rename(old_path, target_name)
					if not os.path.exists(target_name):
						break  # 名前が空いている場合
			else:
				os.rename(old_path, new_path)

			target_path = os.path.join(variables.OUTPUT_PATH, str(new)[4:-2])
			if not os.path.exists(target_path):
				os.mkdir(target_path)
			target_name = os.path.join(target_path, str(new)[2:-2] + "_" + datetimes + EXT_PDF)
			if os.path.exists(target_name):
				dirpath, filename = os.path.split(target_name)
				name, ext = os.path.splitext(filename)
				for i in itertools.count(1):
					newname = '{}_{}{}'.format(name, i, ext)
					target_name = os.path.join(dirpath, newname)
					if not os.path.exists(target_name):
						break  # 名前が空いている場合
			shutil.move(new_path, target_name)
			info_log.info(f"読取結果の保存先({old_path})=>{target_name}")
