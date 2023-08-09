# utf-8
import math
import re
import sys
import traceback

import PySimpleGUI as sg
import numpy as np
import pandas as pd
from cv2 import cv2

import characterRecognition
import rewriteConfig
import variables
from CONST import *

info_log = logging.getLogger(INFO_LOG).getChild(__name__)
info_log.setLevel(logging.INFO)
error_log = logging.getLogger(ERROR_LOG).getChild(__name__)
error_log.setLevel(logging.ERROR)
# 呼び出し元でハンドラが設定されない限りログを出力しない
logging.getLogger(__package__).addHandler(logging.NullHandler())


class MainGUI:
	def __init__(self):
		self.event = None
		self.values = None
		self.col = None
		self.row = None
		self.org_images = None
		self.from_img_for_mark = []
		self.sus_list = []
		self.error_list = []
		self.check_dict = defaultdict(list)
		sg.theme('DarkBlue13')
		sg.SetOptions(element_padding=(0, 0), font=('メイリオ', 10))
		self.form = sg.Window('Character Recognition', resizable=True, grab_anywhere=True, return_keyboard_events=True, finalize=True, location=(0, 0))
		self.first_frame = sg.Frame(BLANK, [
			[sg.Text(GUIDE_MESS, expand_x=True)],
			[sg.Text(BLANK)],
			[sg.Text(f"{READ_FOLDER} :"), sg.InputText(default_text=READ_PATH, key=KEY_READ_FOLDER, expand_x=True), sg.FolderBrowse(target=KEY_READ_FOLDER)],
			[sg.Text(f"{OUTPUT_FOLDER} :"), sg.InputText(default_text=OUTPUT_PATH, key=KEY_OUTPUT_FOLDER, expand_x=True), sg.FolderBrowse(target=KEY_OUTPUT_FOLDER)],
			[sg.Text(BLANK)],
			[sg.Button(DECISION_BTN, key=KEY_OK), sg.Cancel(CANCEL_BTN, key=KEY_CANCEL), sg.Button("設定", key="-info-", pad=(100, 0)), sg.Text(f"バージョン：{VERSION}")],
		], key="-first_frame-", expand_x=True)
		self.image_frame = sg.Frame(BLANK, [[sg.Image("", size=(400, 550), enable_events=True, key=KEY_START_IMG, expand_x=True, expand_y=True)]], pad=(1, 1), key="-image_frame-", expand_x=True, expand_y=True)
		self.result_frame = sg.Frame("認識結果", layout=[[]], key="-result_frame-", expand_x=True, expand_y=True)
		column = [[self.image_frame, self.result_frame]]
		self.layout = [[
			self.first_frame,
			[sg.Column(column, key="-result-", visible=False, expand_x=True, expand_y=True)],
		]]
		self.form.Layout(self.layout)

	def show_org_images(self, _btn_no):
		nums = int(_btn_no[3:-8])
		org_image = self.org_images[nums]
		org_image = cv2.resize(org_image, dsize=None, fx=0.4, fy=0.4)
		byte_org_image = cv2.imencode(EXT_PNG, org_image, params=[int(cv2.IMWRITE_WEBP_QUALITY), 50])[1].tobytes()
		return byte_org_image

	def re_set_result_frame(self, df):
		global now_loc
		global final_df
		now_loc = ["0", "1"]
		path_df = df['PATH']
		df.drop(['PATH'], axis=1, inplace=True)
		if 'sus_nums' in df.columns:
			self.sus_list = df['sus_nums'].values.tolist()
			df.drop(['sus_nums'], axis=1, inplace=True)
		self.sus_list = [[] if sus is None else sus for sus in self.sus_list]

		if 'ERROR' in df.columns:
			self.error_list = df['ERROR'].values.tolist()
			df.drop(["ERROR"], axis=1, inplace=True)
		self.error_list = [[] if error is None else error for error in self.error_list]
		index_list = df.index.values.tolist()
		final_df = pd.DataFrame(data=index_list, columns=['old_filename'])
		final_df = pd.merge(final_df, path_df, how='inner', left_on='old_filename', right_index=True)
		final_df.assign(new_filename=None)
		if 'img_for_mark' in df.columns:
			self.from_img_for_mark = df['img_for_mark'].values.tolist()
		self.org_images = df['PIL_image'].values.tolist()
		# 見出しの設定
		header_list = df.columns.values.tolist()
		header_list = [
			sg.InputText('No', size=(1, 1), pad=(1, 1), disabled_readonly_background_color='AntiqueWhite1', readonly=True, font=2, justification='center', expand_x=True)
			if header == "PIL_image"
			else sg.InputText('読取箇所', size=(35, 1), pad=(2, 1), disabled_readonly_background_color='AntiqueWhite1', readonly=True, font=5, justification='center', expand_x=True)
			if header == "img_for_mark"
			else sg.InputText(f'{str(int(header) - 1)}文字', size=(5, 1), pad=(1, 1), disabled_readonly_background_color='AntiqueWhite1', readonly=True, font=('Gothic', 8), justification='center')
			for header in header_list]

		# pandasのデータを表として加工する処理
		data = df.values.tolist()
		self.row = (len(data[0]))
		self.col = (len(data))
		pre_list = []
		temp_number_1 = 0
		temp_number_2 = 0
		for i1, rows in enumerate(data):
			for i2, cols in enumerate(rows):
				if i2 == 0:
					if not self.error_list or not self.sus_list:
						button_widget = sg.Button(f"{i1 + 1}", size=(5, 1), pad=(1, 1), key=f'btn_{temp_number_1}', enable_events=True, button_color=RED)
						pre_list.append(button_widget)
					elif type(self.error_list[i1]) == list and len(self.sus_list[i1]) >= 1:
						button_widget = sg.Button(f"{i1 + 1}", size=(5, 1), pad=(1, 1), key=f'btn_{temp_number_1}', enable_events=True, button_color=RED)
						pre_list.append(button_widget)
					elif type(self.error_list[i1]) == list:
						button_widget = sg.Button(f"{i1 + 1}", size=(5, 1), pad=(1, 1), key=f'btn_{temp_number_1}', enable_events=True, button_color=RED)
						pre_list.append(button_widget)
					elif len(self.sus_list[i1]) >= 1:
						button_widget = sg.Button(f"{i1 + 1}", size=(5, 1), pad=(1, 1), key=f'btn_{temp_number_1}', enable_events=True, button_color=OCHER)
						pre_list.append(button_widget)
					else:
						button_widget = sg.Button(f"{i1 + 1}", size=(5, 1), pad=(1, 1), key=f'btn_{temp_number_1}', enable_events=True, button_color=LIGHT_BLUE)
						pre_list.append(button_widget)
					temp_number_1 += 1
				elif i2 == self.row - 1:
					if isinstance(cols, np.ndarray):
						byte_img = cv2.imencode(EXT_PNG, cols, params=[int(cv2.IMWRITE_WEBP_QUALITY), 50])[1].tobytes()
						image_widget = sg.Image(source=byte_img, pad=(10, 1), key=f'img_{temp_number_2}', enable_events=True, expand_x=True)
					else:
						image_widget = sg.Text("※読み取れませんでした", pad=(10, 1), expand_x=True, text_color='red')
						# image_widget = sg.Image(source=None, size=(100, 3), pad=(10, 1), key=f'img_{temp_number_2}', enable_events=True, expand_x=True)
					pre_list.append(image_widget)
					temp_number_2 += 1
				else:
					if cols == '' or cols is None:
						text_widget = sg.InputText(default_text=cols, font=5, size=(3, 1), pad=(1, 1), key=f'txt{(str(i1))}_{(str(i2))}', enable_events=True, background_color='red')
						pre_list.append(text_widget)
					elif len(cols) > 1:
						text_widget = sg.InputText(default_text=BLANK, font=5, size=(3, 1), pad=(1, 1), key=f'txt{(str(i1))}_{(str(i2))}', enable_events=True, background_color='red')
						pre_list.append(text_widget)
					elif str(i2 - 1) in str(self.sus_list[i1]):  # ここは文字を赤くするところ
						text_widget = sg.InputText(default_text=cols, font=5, size=(3, 1), pad=(1, 1), key=f'txt{(str(i1))}_{(str(i2))}', enable_events=True, text_color='red')
						pre_list.append(text_widget)
					else:
						text_widget = sg.InputText(default_text=cols, font=5, size=(3, 1), pad=(1, 1), key=f'txt{(str(i1))}_{(str(i2))}', enable_events=True)
						pre_list.append(text_widget)

		num = len(pre_list) / self.col
		display_data_list = []
		# 1行ごとのリストに変換する処理
		for times in range(self.col):
			for i in range(int(num)):
				if i == int(num - 1):
					temp = pre_list[:int(num)]
					del pre_list[:int(num)]
					display_data_list.append(temp)

		lists = [
			header_list,
			[sg.Column(display_data_list, scrollable=True, size=(800, 550), pad=(1, 3), expand_x=True, expand_y=True, key="-scroll-")],
			[
				sg.Output(pad=(1, 1), size=(80, 6), expand_x=True),
				sg.Button('完　了', font=(MEIRYO_FONT, 14), size=(10, 1), key='-complete-', pad=(1, 3), button_color='CadetBlue4', expand_x=True)
			]
		]
		self.form.extend_layout(self.form["-result_frame-"], lists)
		self.form["-result_frame-"].update(visible=True)
		self.form["-first_frame-"].update(visible=False)
		# self.form.extend_layout(self.form["-first_frame-"], [[]])
		self.form.refresh()
		self.form.finalize()

	def set_focus(self, press_key):
		global now_loc
		global scroll_posi
		r2 = now_loc
		left = 0
		right = 1

		if press_key == "Up:38":
			if int(r2[0]) == 0:
				left = 0
				right = int(r2[1])
			else:
				left = int(r2[0]) - 1
				right = int(r2[1])
				if left % 5 == 0:
					# print(scroll_posi)
					self.form["-scroll-"].Widget.canvas.yview_moveto((left / int(self.col)/0.8))  # おまけ
				res = self.show_org_images(f'txt{left}_{right}_Enter')
				self.form[KEY_START_IMG].update(data=res)
		elif press_key == "Left:37":
			if int(r2[1]) == 1:
				left = int(r2[0])
				right = 1
			else:
				left = int(r2[0])
				right = int(r2[1]) - 1
		elif press_key == "Right:39":
			if int(r2[1]) == self.row - 2:
				left = int(r2[0])
				right = self.row - 2
			else:
				left = int(r2[0])
				right = int(r2[1]) + 1
		elif press_key == "Down:40":
			if int(r2[0]) == self.col - 1:
				left = self.col - 1
				right = int(r2[1])
			else:
				left = int(r2[0]) + 1
				right = int(r2[1])
				if left % 5 == 0:
					self.form["-scroll-"].Widget.canvas.yview_moveto((left / int(self.col)/1.2))
					scroll_posi = (left / int(self.col)/1.2)
				res = self.show_org_images(f'txt{left}_{right}_Enter')
				self.form[KEY_START_IMG].update(data=res)

		elif int(r2[0]) > self.col - 1 or int(r2[1]) > self.row - 2:  # 想定以上の場合はスルー
			pass
		new = f'txt{left}_{right}'
		now_loc = r2
		self.form[new].set_focus(True)
		self.form[new].update(select=True)
		# res = self.ShowOrgImages(new+"_Enter")
		# self.form[KEY_START_IMG].update(data=res)

	def main_loop(self):
		global now_loc
		global final_df
		global scroll_posi
		scroll_posi = 0
		while True:
			event, values = self.form.read()
			if event in [sg.WIN_CLOSED, KEY_CANCEL]:
				sys.exit(self.form.close())
			if event == "-info-":
				change_settings()
			elif re.match(r'^btn_', event):
				line_f = "~~~~~~~~~~~~~~~~"
				if type(self.error_list[int(event[4:])]) == list:
					print(self.error_list[int(event[4:])])
					print(line_f)
				elif math.isnan(self.error_list[int(event[4:])]) and len(self.sus_list[int(event[4:])]) < 1:
					print("エラーなしです")
					print(line_f)
				elif math.isnan(self.error_list[int(event[4:])]) and len(self.sus_list[int(event[4:])]) >= 1:
					print(f"{self.sus_list[int(event[4:])][0] + 1}文字目の数字を確認してください")
					print(line_f)
				elif type(self.error_list[int(event[4:])]) == list and len(self.sus_list[int(event[4:])]) < 1:
					print(self.error_list[int(event[4:])])
					print(line_f)
				else:
					print(self.error_list[int(event[4:])])
					print(f"{self.sus_list[int(event[4:])][0] + 1}文字目の数字を確認してください")
					print(line_f)
			elif "_Enter" in event:
				res = self.show_org_images(event)
				self.form[KEY_START_IMG].update(data=res)
			elif event in ["Right:39", "Left:37", "Up:38", "Down:40"]:
				press_key = event
				self.set_focus(press_key)
			elif re.match(r'^txt', event):
				patter = r'^(\d{1})?$'
				patter2 = r'\d{1}'
				if not re.match(patter, values.get(event)):
					sg.popup_error('入力は「半角数字」のみ可能です。', title='入力エラー')
					self.form[event].update("")
				if re.match(patter2, values.get(event)):
					target = 'txt'
					idx = event.find(target)
					r = event[idx + len(target):]
					r2 = r.split('_')
					left = 0
					right = 1
					if values.get(event) != self.check_dict[event][0]:
						self.form[event].update(background_color=LIGHT_GREEN, text_color='green')
						if int(r2[1]) < self.row - 2:  # 8文字目以外の場合、右にズレる
							left = int(r2[0])
							right = int(r2[1]) + 1
						elif int(r2[1]) == self.row - 2 and int(r2[0]) == self.col - 1:  # 最集団の場合はループ
							left = int(r2[0])
							right = int(r2[1])
						elif int(r2[1]) == self.row - 2:  # ８文字目の場合、下の段にズレる
							left = int(r2[0]) + 1
							right = 1
							res = self.show_org_images(f'txt{left}_{right}_Enter')
							self.form[KEY_START_IMG].update(data=res)
						# エンター
						elif int(r2[0]) > self.col - 1 or int(r2[1]) > self.row - 2:  # 想定以上の場合はスルー
							pass
						new = f'txt{left}_{right}'
						now_loc = r2
						self.form[new].set_focus(True)
						self.form[new].update(select=True)

					else:
						now_loc = r2
			# elif ord(event[0]) == 13:
			# 	try:
			# 		left = int(now_loc[0]) + 1
			# 		right = 1
			# 		if int(now_loc[0]) == self.col - 1:  # 最終段の場合はパス
			# 			pass
			# 		else:
			# 			new = f'txt{left}_{right}'
			# 			self.form[new].set_focus(True)
			# 			self.form[new].update(select=True)
			# 			res = self.ShowOrgImages(new + '_Enter')
			# 			now_loc = [int(now_loc[0]) + 1, 1]
			# 			self.form[KEY_START_IMG].update(data=res)
			# 	except KeyError:
			# 		pass
			# 	except ValueError:
			# 		print("最後の行です")
			# 		pass
			# 	except Exception as e:
			# 		print(repr(e))
			# 		pass
			elif event == '-complete-':
				button_widget = values
				file_dict = defaultdict(list)
				yellow_view = 0
				for i1 in range(self.col):
					for i2 in range(self.row):
						if button_widget.get(f'txt{i1}_{i2}') is None:
							pass
						elif button_widget.get(f'txt{i1}_{i2}') == '':
							self.form[f'txt{i1}_{i2}'].update(background_color='yellow', text_color='white')
							yellow_view += 1
						else:
							file_dict[i1].append(button_widget.get(f'txt{i1}_{i2}'))

				final_list = defaultdict(list)
				for k, v in file_dict.items():
					if ''.join([str(n) for n in v]) == "":
						pass
					else:
						v.insert(1, '_')
						filename = ''.join([str(n) for n in v])
						final_list[k].append(filename)
				file_list2 = final_list.copy()
				for i, v in file_list2.items():
					if len(v[0]) < READ_NUMBERS + 1:
						del final_list[i]
				if len(final_list) == 0:
					sg.popup_error(NO_MOVE_MESS[1], title=NO_MOVE_MESS[0])
					info_log.info(NO_MOVE_MESS[1])
					sys.exit()
				for k, v in final_list.items():
					final_df.loc[k, 'new_filename'] = v

				final_df.drop(['old_filename'], axis=1, inplace=True)
				list_rename = final_df.values.tolist()
				self.form.close()
				return list_rename
			elif event == KEY_OK:

				if values[KEY_READ_FOLDER] != READ_PATH:
					Read_Path = values[KEY_READ_FOLDER]
					rewriteConfig.rewrite_ini("ReadPath", values[KEY_READ_FOLDER])
					info_log.info(f"{READ_FOLDER} => {Read_Path}\n{REWRITE_MESS}")
					variables.READ_PATH = Read_Path
				if values[KEY_OUTPUT_FOLDER] != OUTPUT_PATH:
					output_path = values[KEY_OUTPUT_FOLDER]
					rewriteConfig.rewrite_ini("OutputPath", values[KEY_OUTPUT_FOLDER])
					info_log.info(f"{OUTPUT_FOLDER} => {output_path}\n{REWRITE_MESS}")
					variables.OUTPUT_PATH = output_path
				re_df = characterRecognition.start_ocr()
				self.re_set_result_frame(df=re_df)
				self.form["-result-"].update(visible=True)
				print('読取画像と比較して認識結果に差異がある場合は、左記のテキストボックスで編集してください。\n・赤文字は認識率が95%以下の数字です。\n・読取結果を削除したい場合は、テキストボックスの数字をすべて削除して空にしてください。\n')
				event, values = self.form.read()
				# self.form.move_to_center()
				for i1 in range(self.col):
					for i2 in range(1, 9):
						try:
							self.check_dict[f'txt{i1}_{i2}'].append(values.get(f'txt{i1}_{i2}'))
							self.form[f'txt{i1}_{i2}'].bind("<Enter>", '_Enter')
						except AttributeError as ae:
							error_log.error(traceback.format_exc())
				self.form["txt0_1"].set_focus(True)


def change_settings():
	sg.theme("Reds")
	form = sg.Window('設定画面', resizable=True, finalize=True, keep_on_top=True)
	layout = [
		[sg.Text("01"), sg.Text("読取枚数：", size=(40, 0), justification="right"), sg.InputText(default_text=READ_AMOUNT, key="1.", size=(10, 0))],
		[sg.Text("  "), sg.Text("認識エリア：", size=(40, 0), justification="right")],
		[sg.Text("02"), sg.Text("上：", size=(40, 0), justification="right"), sg.InputText(default_text=TRIM_TOP, key="2.", size=(10, 0))],
		[sg.Text("03"), sg.Text("下：", size=(40, 0), justification="right"), sg.InputText(default_text=TRIM_BOTTOM, key="3.", size=(10, 0))],
		[sg.Text("04"), sg.Text("右：", size=(40, 0), justification="right"), sg.InputText(default_text=TRIM_RIGHT, key="4.", size=(10, 0))],
		[sg.Text("05"), sg.Text("左：", size=(40, 0), justification="right"), sg.InputText(default_text=TRIM_LEFT, key="5.", size=(10, 0))],
		[sg.Text("06"), sg.Text("読取文字数：", size=(40, 0), justification="right"), sg.InputText(default_text=READ_NUMBERS, readonly=True, size=(10, 0))],
		[sg.Text("07"), sg.Text("DPI：", size=(40, 0), justification="right"), sg.InputText(default_text=DPI, key="6.", size=(10, 0))],
		[sg.Text("08"), sg.Text("画像を水平にするための許容誤差：", size=(40, 0), justification="right"), sg.InputText(default_text=ALLOWABLE_DIFF, key="7.", size=(10, 0))],
		[sg.Text("09"), sg.Text("枠線を判別するための閾値（枠線の長さ）：", size=(40, 0), justification="right"), sg.InputText(default_text=THRESH_SPIKE, key="8.", size=(10, 0))],
		[sg.Text("10"), sg.Text("数字と判別するための最小の大きさ：", size=(40, 0), justification="right"), sg.InputText(default_text=CHARACTER_SIZE, key="9.", size=(10, 0))],
		[sg.Text("11"), sg.Text("空白があることを判別するための最小の文字間距離：", size=(40, 0), justification="right"), sg.InputText(default_text=BLANK_DISTANCE, key="10.", size=(10, 0))],
		[sg.Text("12"), sg.Text("水平線とみなす閾値：", size=(40, 0), justification="right"), sg.InputText(default_text=HORIZON_THREAD, key="11.", size=(10, 0))],
		[sg.Submit("変更", key="-change-"), sg.Cancel("戻る", key="-return-"), sg.Button("初期値に戻す", key="-default-", pad=(100, 0))]
	]
	form.Layout(layout)
	while True:
		event, values = form.read()
		if event == sg.WIN_CLOSED or event == "-return-":
			info_log.info(CLOSE_MESS)
			break
		elif event == "-change-":
			change_dict = {
				"read_amount": str(values["1."]),
				"trim_top": str(values["2."]),
				"trim_bottom": str(values["3."]),
				"trim_right": str(values["4."]),
				"trim_left": str(values["5."]),
				"dpi": str(values["6."]),
				"allowable_diff": str(values["7."]),
				"thresh_spike": str(values["8."]),
				"character_size": str(values["9."]),
				"blank_distance": str(values["10."]),
				"horizon_thread": str(values["11."])}
			rewriteConfig.change_settings(change_dict)
			break
		elif event == "-default-":
			rewriteConfig.reset_default()
			break
	form.close()
