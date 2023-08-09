# utf-8
"""
OCR読取の概要
①PDFファイルをイメージファイルに変換する
　popperという外部のアプリとPDF2imageというサードパーティー製のライブラリを使用
②イメージ化した画像の「文字認識する箇所」をトリミングする、それと同時に画像の形式をnd.arrayに変換する
③以下文字認識するための下準備
	０１．傾き角を求める
	０２．傾きを補正するために、一旦Pillowに変換する
	０３．Pillow画像の傾きを修正する
	０４．傾きを修正した画像をもう一度トリミングする
	０５．修正した傾き画像をnd.arrayに変換する
	０６．縦線を除去する処理（利用者コードの記入欄）
	０７．水平線を除去する処理（利用者コードの記入欄）
		※利用者コードの記入欄を削除する理由
		・矩形を認識して内部の数字を読み取るという方法も試したが、次の場合にエラーとなる
		・枠線を飛び越えて数字が書かれている場合
		・そのため、枠線を削除して文字だけにするのに処理している
	０８．傾きを修正し、枠線を除去した数字だけの画像から文字を認識する
		・画像を2極化
		・輪郭を抽出（指定した大きさ以上のものを抽出する（小さいものは除外する））
		・抽出した輪郭をⅹ軸の昇順に並び替える
		・空白の文字がないか確認する
			・輪郭同士のⅹ軸の間隔が一定以上空いていたら空白と判断
		・想定している輪郭の個数分（ここでは8文字）の画像と輪郭にマーキングした画像を返す
		・一文字づつの画像をさらに認識しやすいように加工する
			*画像の輪郭を抽出する
			*画像の最大面積の輪郭以外は背景色でうめる
			*輪郭の中で領域面積が最大のものを取得
			*モルフォロジー変換
			*矩形の縦横比を保ったままリサイズする
			*重心を画像のセンターへ移動する
④文字数の確認を行う（ここでは8文字）
⑤文字の認識作業開始（数字一文字づつ）
⑥認識結果をもとに、認識率が95％以下のものを抽出する
⑦認識結果をデータフレームにして情報を整理する
"""

import glob
import math
import operator
import os
import sys
import traceback
from pathlib import Path
from typing import Tuple, List, Dict

import PIL
import PySimpleGUI as sg  # ライブラリの読み込み
import numpy as np
import pandas as pd
from PIL import Image, PngImagePlugin  # 画像データ用
from cv2 import cv2

from numpy import array
from pdf2image import convert_from_path

import rewriteConfig
from CONST import *
import variables

info_log = logging.getLogger(INFO_LOG)
info_log.setLevel(logging.INFO)
error_log = logging.getLogger(ERROR_LOG)
error_log.setLevel(logging.ERROR)
# 呼び出し元でハンドラが設定されない限りログを出力しない
logging.getLogger(__package__).addHandler(logging.NullHandler())


def show_preview(_name, _img):
	"""
	テスト時のみプレビューを表示する
	:param _name:
	:param _img:
	:return:
	"""
	if variables.test_flag:
		cv2.imshow(_name, _img)
		cv2.waitKey()
		cv2.destroyAllWindows()


def pdf_change_image():
	"""
	PDFをイメージファイルに変換する処理
	"Poppler"という外部アプリを使用している
	"""
	# poppler/binを環境変数PATHに追加する
	poppler_dir = os.path.join(variables.ab_path, POPPER_BIN)
	os.environ["PATH"] += os.pathsep + str(poppler_dir)
	# PDFファイルのパス
	pdf_paths = glob.glob(variables.READ_PATH + SUBFORDER_SEARCH + EXT_PDF)

	while 0 == len(pdf_paths):
		res = sg.popup_ok_cancel(NO_PDF_MESS % variables.READ_PATH, title=NO_PDF_TITLE)
		if res == 'OK':
			sec_read_path = sg.popup_get_folder(title=SEC_SEARCH_MESS[0], message=SEC_SEARCH_MESS[1])
			if sec_read_path is None:
				sys.exit(info_log.info(SEC_EXIT_MESS))
			if sec_read_path != BLANK:
				pdf_paths = glob.glob(sec_read_path + SUBFORDER_SEARCH + EXT_PDF)
			variables.READ_PATH = sec_read_path
			rewriteConfig.rewrite_ini("ReadPath", sec_read_path)
		else:
			sys.exit(info_log.info(SEC_EXIT_MESS))
	read_pdfs = len(pdf_paths)
	if read_pdfs > int(READ_AMOUNT):
		del pdf_paths[int(READ_AMOUNT):]
		read_pdfs = len(pdf_paths)
	info_log.info(f"読取PDF数={read_pdfs}")
	for i, pdf_path in enumerate(pdf_paths):
		name = Path(pdf_path).stem
		pages: PIL.PngImagePlugin.PngImageFile = convert_from_path(str(pdf_path), dpi=DPI)
		all_dict[name].append(np.array(pages[0], dtype=np.uint8))
		all_dict[name].append(pdf_path)
		re = sg.OneLineProgressMeter(title='画像変換中...', current_value=i + 1, max_value=read_pdfs, keep_on_top=True)
		if not re:
			info_log.info("キャンセルされました。")
			sys.exit()


class TrimImage:
	"""
	指定の場所だけを抽出する処理
	"""

	def __init__(self, _img: PIL.PngImagePlugin.PngImageFile or np.ndarray, _class_type: str = None) -> None:
		"""
		初期化処理
		* PILかOpenCVによって型が違うのでそれぞれ処理を変更している
		:param _img: pil_image 又は np.ndarray
		:param _class_type: 文字列のNumPyかPillow
		:return None
		"""
		self.img = _img
		self.type = self.check_image_type()
		self.want = _class_type
		if self.type == PILLOW and (self.want == PILLOW or self.want is None):
			self.rough_trim = self.roughly_trim_image()
			self.correct_trim = self.correctly_trim_image()
		elif self.type == NUMPY and self.want == PILLOW:
			self.img = self.cv_2_pil()
			self.rough_trim = self.roughly_trim_image()
			self.correct_trim = self.correctly_trim_image()
		elif self.type == PILLOW and self.want == NUMPY:
			self.rough_trim = self.roughly_trim_image()
			self.correct_trim = self.correctly_trim_image()
			self.rough_trim = self.pil_2_cv(self.rough_trim)
			self.correct_trim = self.pil_2_cv(self.correct_trim)
		elif self.type == NUMPY and (self.want == NUMPY or self.want is None):
			self.img = self.cv_2_pil()
			self.rough_trim = self.roughly_trim_image()
			self.correct_trim = self.correctly_trim_image()
			self.rough_trim = self.pil_2_cv(self.rough_trim)
			self.correct_trim = self.pil_2_cv(self.correct_trim)

	def roughly_trim_image(self) -> PIL.PngImagePlugin.PngImageFile:
		"""
		おおざっぱにトリミングする処理
		* 右端は読取る画像の右端を指定している。
		:return im_crop: おおざっぱにトリミングした画像
		"""
		w, h = self.img.size
		im_crop = self.img.crop((TRIM_LEFT, TRIM_TOP, w - 10, TRIM_BOTTOM))
		return im_crop

	# トリミングする関数
	def correctly_trim_image(self) -> PIL.PngImagePlugin.PngImageFile:
		"""
		おおざっぱにトリミングした画像をさらに絞り込んでトリミングする処理
		:return trim_image_file: 正確にトリミングした画像
		"""
		rgb_image = self.rough_trim.convert('RGB')
		size = rgb_image.size
		t_pix = 100000000
		b_pix = -1
		l_pix = -1
		r_pix = -1
		# ピクセル操作
		for x in range(size[0]):
			for y in range(size[1]):
				r, g, b = rgb_image.getpixel((x, y))
				rr, rg, rb = rgb_image.getpixel((size[0] - x - 1, size[1] - y - 1))
				# 色付きのピクセルかどうか（白もしくは白に近しい色を切り抜くため）
				if (r + g + b) < 600:
					if l_pix == -1:
						l_pix = x
					if y < t_pix:
						t_pix = y
				if (rr + rg + rb) < 600:
					if r_pix == -1:
						r_pix = size[0] - x
					if size[1] - y > b_pix:
						b_pix = size[1] - y
		trim_image_file = self.rough_trim.crop((l_pix, t_pix, r_pix, b_pix))  # トリミング
		return trim_image_file

	def pil_2_cv(self, _img: PIL.PngImagePlugin.PngImageFile = None) -> np.ndarray:
		"""
		PIL型 -> OpenCV型
		:param _img: PIL型の画像
		:return : OpenCV型の画像
		"""
		if _img is not None:
			new_image = np.array(_img, dtype=np.uint8)
		else:
			new_image = np.array(self.img, dtype=np.uint8)
		if new_image.ndim == 2:  # モノクロ
			pass
		elif new_image.shape[2] == 3:  # カラー
			new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
		elif new_image.shape[2] == 4:  # 透過
			new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
		return new_image

	def cv_2_pil(self, _img: np.ndarray = None) -> PIL.PngImagePlugin.PngImageFile:
		"""
		OpenCV型 -> PIL型
		:param _img: OpenCV型の画像
		:return: PIL型の画像
		"""
		if _img is not None:
			new_image = _img.copy()
		else:
			new_image = self.img.copy()
		if new_image.ndim == 2:  # モノクロ
			pass
		elif new_image.shape[2] == 3:  # カラー
			new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
		elif new_image.shape[2] == 4:  # 透過
			new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
		new_image = Image.fromarray(new_image)
		return new_image

	def check_image_type(self) -> str:
		"""
		画像データがNumPyか、Pillowかを調べる
		:return: 文字列のNumPyかPillowを返す
		"""
		if isinstance(self.img, np.ndarray):
			return NUMPY
		elif isinstance(self.img, Image.Image):
			return PILLOW


# noinspection PyMethodMayBeStatic
class NextPredict:
	"""
	画像認識する前の処理
	"""

	def __init__(self, _img: np.ndarray):
		self.img = _img
		self.nes_img = self.preprocess()

	def fill_unnecessary_area(self, _img: np.ndarray, _cntrs: Tuple) -> np.ndarray:
		"""
		img内の輪郭cntrsを背景色で埋める
		:return: 背景色で埋められた画像
		"""
		trim_img = ''
		for c in _cntrs:
			x, y, w, h = cv2.boundingRect(c)
			trim_img = cv2.rectangle(_img, (x, y), (x + w, y + h), GREEN_COLOR, 0)
		return trim_img

	def padding_position(self, x: int, y: int, w: int, h: int, pad: int) -> Tuple[int, int, int, int]:
		"""
		# 抽出した矩形のパラメータ(x, y, w, h)にpad分余白を持たせる(Qiita [機械学習のためのOpenCV入門]より)
		外接矩形
		:param x:
		:param y:
		:param w:
		:param h:
		:param pad:
		:returns: 余白を持たせた外接矩形の情報
		"""
		return x - pad, y - pad, w + pad * 2, h + pad * 2

	def morph_transformation(self, _img: np.ndarray) -> np.ndarray:
		"""
		モルフォロジー処理には収縮と膨張の2つの基本処理がある。
		画像中のオブジェクトに対して、収束や膨張を行うと、オブジェクトの周りあるいはオブジェクトの中に含まれているノイズを除去できる。
		:param _img: 画像
		:return: ノイズ除去された画像
		"""
		kernel_1 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
		ret_img1 = cv2.dilate(_img, kernel_1, iterations=2)  # 膨張
		kernel_2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
		ret_img2 = cv2.erode(ret_img1, kernel_2)  # 収縮
		return ret_img2

	def get_maxrect_size(self, w: int, h: int, side_length: int) -> Tuple[int, int]:
		"""
		輪郭抽出した矩形の縦横比を変えない最大の辺の長さ(横, 縦)を返す
		:param w: 幅
		:param h: 高さ
		:param side_length: 変更する幅
		:return: 最大の辺の長さ(横, 縦)
		"""
		size = round(side_length * 0.75)
		aspect_ratio = w / h
		if aspect_ratio >= 1:
			return size, round(size / aspect_ratio)
		else:
			return round(size * aspect_ratio), size

	def move_to_center(self, _img: np.ndarray, new_size: Tuple[int, int]) -> np.ndarray:
		"""
		画像を中心に移動させる
		:param _img: 入力画像
		:param new_size: リサイズするサイズ
		:return: リサイズ後の画像
		"""
		m = cv2.moments(_img)
		# 重心
		cx = int(m['m10'] / m['m00'])
		cy = int(m['m01'] / m['m00'])
		# 移動量の計算
		tx = new_size[1] / 2 - cx
		ty = new_size[0] / 2 - cy
		# x軸方向にtx, y軸方向にty平行移動させる
		m = np.float32([[1, 0, tx], [0, 1, ty]])
		dst = cv2.warpAffine(_img, m, new_size)
		return dst

	def preprocess(self, blank=253, min_size=300, padding=1, new_size=(28, 28)) -> np.ndarray or None:
		"""
		画像認識ライブラリに投げる前の処理
		:param blank: デフォルト値
		:param min_size: 使用していない
		:param padding: パディング
		:param new_size: この数字は固定（1文字のピクセルサイズ）
		:return: 白黒の1文字だけの画像 または None（エラーの場合）
		"""
		img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
		if np.sum(img_gray) / img_gray.size >= blank:
			# 白紙の場合は削除
			return None
		# ----- 画像の輪郭を抽出する -----
		img_blur1 = cv2.GaussianBlur(img_gray, (11, 11), 0)
		img_inv = cv2.threshold(img_blur1, 245, 255, cv2.THRESH_BINARY_INV)[1]
		contours, hierarchy = cv2.findContours(img_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		# ----- 最大面積の輪郭以外は背景色で埋める -----
		img_blur2 = cv2.GaussianBlur(img_gray, (5, 5), 0)
		img_inv2 = cv2.threshold(img_blur2, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
		max_area_idx = np.argmax([cv2.contourArea(c) for c in contours])  # 輪郭の中での領域面積が最大のものを取得
		max_area = list(contours).pop(max_area_idx)  # 最大面積の輪郭をcontoursから取り出して削除しておく
		tmp_img = self.fill_unnecessary_area(img_inv2, contours)
		x, y, w, h = cv2.boundingRect(max_area)
		if x >= padding and y >= padding:
			x, y, w, h = self.padding_position(x, y, w, h, padding)
		# ----- モルフォロジー変換 -----
		tmp_img = self.morph_transformation(tmp_img)
		# ----- 矩形の縦横比を保ったままリサイズする -----
		cropped = tmp_img[y:(y + h), x:(x + w)]
		new_w, new_h = self.get_maxrect_size(w, h, new_size[0])
		new_cropped = cv2.resize(cropped, (new_w, new_h))
		# ----- 重心を画像のセンターへ移動 -----
		dst_img: np.ndarray = self.move_to_center(new_cropped, new_size)
		return dst_img


# noinspection PyMethodMayBeStatic
class Predict:
	"""
	トリミングした画像から、数字をひとつづつ検出するための処理
	* ここでの画像はすべて`np.ndarray`
	"""

	def __init__(self, _img: np.ndarray) -> None:
		"""
		初期化処理
		:param _img: 正確にトリミングした画像
		"""
		self.img = _img
		self.copy_img = self.img.copy()
		self.gray_img = cv2.cvtColor(self.copy_img, cv2.COLOR_BGR2GRAY)
		self.reversed_gray = cv2.bitwise_not(self.gray_img)
		self.img_blur = cv2.GaussianBlur(self.reversed_gray, (5, 5), 3)
		self.re_img = self.predict_main()

	def replacement_img(self, _img: np.ndarray) -> None:
		"""
		self.copy_imgの内容を置き換える（上書きするための）処理
		:param _img: 置き換えたい画像
		:return: None
		"""
		self.copy_img = _img
		self.gray_img = cv2.cvtColor(self.copy_img, cv2.COLOR_BGR2GRAY)
		self.reversed_gray = cv2.bitwise_not(self.gray_img)
		self.img_blur = cv2.GaussianBlur(self.reversed_gray, (5, 5), 3)

	def auto_canny(self) -> np.ndarray:
		"""[md]
		画像の輝度差が大きい部分を輪郭として検出するための処理 [参考](https://qiita.com/kotai2003/items/662c33c15915f2a8517e)
		1. img_blurから中央値を算出
		2. min_val : 0と(1.0 - sigma) * 中央値)の間の最大値
		3. max_val : 255と(1.0 + sigma) * 中央値)の間の最大値
		* sigmaは0.33とする。
		:return:Cannyした画像
		"""
		med_val = np.median(self.img_blur)  # 1
		sigma = 0.33  # 0.33
		min_val = int(max(0, (1.0 - sigma) * med_val))  # 2
		max_val = int(max(255, (1.0 + sigma) * med_val))  # 3
		img_edge = cv2.Canny(self.img_blur, threshold1=min_val, threshold2=max_val)
		return img_edge

	def get_degree(self) -> int or float:
		"""
		画像を水平にするため、回転角度を求める処理
		参考：https://reverent.hateblo.jp/entry/2017/01/11/112726
		:return: 回転角度
		"""
		global HORIZONTAL
		edges = self.auto_canny()
		lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=100, minLineLength=120, maxLineGap=20)
		sum_arg = 0
		count = 0
		for line in lines:
			for x1, y1, x2, y2 in line:
				# line_img = cv2.line(self.img, (x1, y1), (x2, y2), (0, 0, 255), 3)
				# Preview(line_img)
				arg = math.degrees(math.atan2((y2 - y1), (x2 - x1)))
				HORIZONTAL = 0
				diff = ALLOWABLE_DIFF  # 許容誤差 -> -20 - +20 を本来の水平線と考える
				if HORIZONTAL - diff < arg < HORIZONTAL + diff:
					sum_arg += arg
					count += 1
		if count == 0:
			return HORIZONTAL
		else:
			return (sum_arg / count) - HORIZONTAL

	def erase_vertical(self) -> np.ndarray:
		"""
		水平線の抽出に画像バッファ生成
		縦線を抽出する処理（https://kyudy.hatenablog.com/entry/2019/10/26/141330）
		:return:
		"""
		img = self.copy_img.copy()
		_, img_th = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_OTSU)
		# reverse white and black for dilate and erode
		img_th = 255 - img_th
		# vertical kernel to connect dot line to solid line
		kernel = np.zeros((5, 5), np.uint8)
		kernel[:, 2] = 1
		img_th = cv2.dilate(img_th, kernel, iterations=2)
		img_th = cv2.erode(img_th, kernel, iterations=2)
		vp = np.sum((img_th != 0).astype(np.uint8), axis=0)
		loc_x_spike = np.where(vp > THRESH_SPIKE)
		# draw vertical lines
		for x in loc_x_spike[0]:
			cv2.line(img, (x, 0), (x, img.shape[0]), WHITE_COLOR, thickness=2)
		# if test_flag:
		# 	_, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4)
		# 	ax1.imshow(self.copy_img)
		# 	ax2.scatter(range(len(vp)), vp)
		show_preview(VERTICAL_ERASE, img_th)
		# 	ax3.imshow(img_th)
		# 	ax4.imshow(img)
		# 	plt.show()
		return img

	def erase_horizon(self) -> np.ndarray:
		"""
		# 横線を消す処理、ハフ変換で横線だけを検出（https://teratail.com/questions/251573）
		# 直線を描画する。
		:return: 横線を消した画像
		"""

		def draw_line(_img, _theta, _rho):
			h, w = _img.shape[:2]
			if np.isclose(np.sin(_theta), 0):
				x1, y1 = _rho, 0
				x2, y2 = _rho, h
			else:
				calc_y = lambda x: _rho / np.sin(_theta) - x * np.cos(_theta) / np.sin(_theta)
				x1, y1 = 0, int(calc_y(0))
				x2, y2 = w, int(calc_y(w))
			cv2.line(img=_img, pt1=(x1, y1), pt2=(x2, y2), color=WHITE_COLOR, thickness=6)  # thicknessは線の太さ

		img = self.copy_img.copy()
		edges = self.auto_canny()
		# ハフ変換で直線検出する。
		lines = cv2.HoughLines(image=edges, rho=1, theta=np.pi / 180, threshold=HORIZON_THREAD)  #
		lines = lines.squeeze(axis=1)
		# 直線の角度がほぼ90°(-85° ~ 95°) のものだけ抽出する。
		lines = list(filter(lambda x: abs(x[1] - np.pi / 2) <= np.deg2rad(5), lines))
		for rho, theta in lines:
			draw_line(img, theta, rho)
		show_preview("test", img)
		return img

	def contour_extraction(self) -> Tuple[Dict[int, np.ndarray], np.ndarray or None]:
		"""
		# [数字を認識](https://ailog.site/2019/08/17/ocr1/)
		画像の読み込み
		グレイスケールに変換しぼかした上で二値化する
		:return num_index_oneimage: {インデックス: 切り取った数字のimg}
		:return img_for_mark: 8桁の数字に対して読取結果をマーキングした画像
		"""

		def format_file_name(r, c) -> str:
			"""
			数字ひとつづつに名前を付ける処理
			:param r:
			:param c:
			:return: ファイル名
			"""
			return 'num' + str(r).rjust(2, '0') + '_' + str(c) + EXT_PNG

		thresh = cv2.adaptiveThreshold(self.img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 29, -2)
		# 輪郭を抽出
		contours = cv2.findContours(
			thresh,
			cv2.RETR_EXTERNAL,  # ☆☆☆領域の一番外側だけを検出☆☆☆
			cv2.CHAIN_APPROX_SIMPLE)[0]
		rectangles = []  # 抽出する数字の輪郭長方形の情報をあつめる
		# 抽出した領域を繰り返し処理する
		for cnt in contours:
			rectangle_points = cv2.boundingRect(cnt)
			x, y, w, h = cv2.boundingRect(cnt)
			if h < CHARACTER_SIZE:  # 小さすぎるのは飛ばす
				continue
			# print(i, (x, y, w, h))
			# cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
			rectangles.append(rectangle_points)
		rectangles = sorted(rectangles, key=operator.itemgetter(0))
		num_index_oneimage: Dict[int, np.ndarray] = {}  # NOTE {インデックス: 切り取った数字のimg}
		img_for_mark = self.copy_img.copy()  # NOTE 8桁の数字に対して読取結果をマーキングした画像
		pre_left = 0  # ひとつ前の左の数字
		flag = False  # 間の空白を検知するためのフラグ
		if len(rectangles) != READ_NUMBERS:
			color = BLUE_COLOR
		else:
			color = RED_COLOR
		for i, rect in enumerate(rectangles):
			if flag:
				i = i + 1
			left = rect[0]
			top = rect[1]
			right = rect[0] + rect[2]
			bottom = rect[1] + rect[3]
			if i >= 2 and left - pre_left > BLANK_DISTANCE:  # 空白の距離
				num_index_oneimage[i] = None
				distance = int((left + pre_left) / 2)
				cv2.rectangle(img_for_mark, (distance, top), (distance + 20, bottom), BLUE_COLOR, -1)
				target = self.copy_img[top:bottom, left:right]
				show_preview(BLANK_SPACE, target)
				counters = [array([[left, top], [left, bottom], [right, top], [right, bottom]])]
				draw_point = (left, top)
				cv2.drawContours(img_for_mark, counters, -1, color, 1)
				cv2.putText(img_for_mark, str(i), draw_point, cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
				# set_file_name = format_file_name(1, i)
				one_number_img: np.ndarray = NextPredict(target).nes_img
				num_index_oneimage[i + 1] = one_number_img
				flag = True
			else:
				target = self.copy_img[top:bottom, left:right]
				show_preview(NO_BLANK_SPACE, target)
				counters = [array([[left, top], [left, bottom], [right, top], [right, bottom]])]
				draw_point = (left, top)
				cv2.drawContours(img_for_mark, counters, -1, color, 1)
				cv2.putText(img_for_mark, str(i), draw_point, cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
				# set_file_name = format_file_name(1, i)
				one_number_img: np.ndarray = NextPredict(target).nes_img
				num_index_oneimage[i] = one_number_img
			pre_left = left
		return num_index_oneimage, img_for_mark

	def second_correctly_trim_image(self, _img: np.ndarray) -> np.ndarray:
		"""
		もう一度トリミング処理を行う
		:param _img: 画像
		:return: トリミングした画像
		"""
		rgb_image = _img.convert('RGB')
		size = rgb_image.size
		t_pix = 100000000
		b_pix = -1
		l_pix = -1
		r_pix = -1
		# ピクセル操作
		for x in range(size[0]):
			for y in range(size[1]):
				r, g, b = rgb_image.getpixel((x, y))
				rr, rg, rb = rgb_image.getpixel((size[0] - x - 1, size[1] - y - 1))
				# 色付きのピクセルかどうか（白もしくは白に近しい色を切り抜くため）
				if (r + g + b) < 600:
					if l_pix == -1:
						l_pix = x
					if y < t_pix:
						t_pix = y
				if (rr + rg + rb) < 600:
					if r_pix == -1:
						r_pix = size[0] - x
					if size[1] - y > b_pix:
						b_pix = size[1] - y
		trim_image_file = _img.crop((l_pix, t_pix, r_pix, b_pix))  # トリミング
		return trim_image_file

	def predict_main(self) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
		"""
		# このクラスののメイン処理
		:returns [files, img]: {インデックス: 切り取った数字のimg}, 8桁の数字に対して読取結果をマーキングした画像
		"""
		arg = self.get_degree()
		new = Image.fromarray(self.img)
		pil_new = new.rotate(angle=arg, fillcolor=WHITE_COLOR, resample=Image.BICUBIC)
		t2_img = self.second_correctly_trim_image(pil_new)
		rotate_img = np.array(t2_img, dtype=np.uint8)
		self.replacement_img(rotate_img)
		img = self.erase_vertical()
		self.replacement_img(img)
		img = self.erase_horizon()
		self.replacement_img(img)
		files, img = self.contour_extraction()
		return files, img


def predict_nnc(img: np.ndarray or None) -> None or Tuple[List[int], int]:
	"""
	nnablaを用いた機械学習の学習データと照合して予測結果を返す
	:param img: 数字一文字の画像　下処理済み
	:return predict_y, predict_number : 予測した数字と予測率
	"""
	if img is None:  # 読取結果がエラーの場合はNoneがくるので、Noneで返す
		return None
	if img.size != 0:  # np.ndarrayがちゃんとあるときだけ処理をおこなう
		from nnabla.utils import nnp_graph  # ここで読んでいるのは、先に呼ぶとロガーが穢されるため
		nnp = nnp_graph.NnpLoader(NNP_PATH)
		graph = nnp.get_network('MainRuntime', batch_size=1)
		x = list(graph.inputs.values())[0]
		y = list(graph.outputs.values())[0]
		x.d = img.reshape(1, 1, 28, 28) * 1.0 / 255
		y.forward()
		predict_y = y.d
		predict_number = np.argmax(y.d)
		return predict_y, predict_number
	return None  # 上記以外のエラーの場合は`None`を返す


def start_ocr():
	try:
		pdf_change_image()
		org_name: str  # NOTE 読取ったPDFファイルの名前(拡張子なし)
		pil_img: PIL.PngImagePlugin.PngImageFile  # NOTE 読取ったPDFファイルをイメージファイルに変換したもの
		for ind, content in enumerate(all_dict.items()):
			org_name = content[0]
			pil_img = content[1]
			t_img: np.ndarray = TrimImage(pil_img[0], NUMPY)
			show_preview(TRIMMED_IMAGE, t_img.correct_trim)
			re = sg.OneLineProgressMeter(
				title='認識処理中...',
				bar_color="red",
				current_value=ind + 1,
				max_value=len(all_dict),
				keep_on_top=True
			)
			if not re:
				info_log.info("キャンセルされました。")
				sys.exit()
			try:
				re_num_index_oneimage: Dict[int, np.ndarray]  # NOTE {インデックス: 切り取った数字のimg}
				re_img_for_mark: np.ndarray  # NOTE 8桁の数字に対して読取結果をマーキングした画像
				re_num_index_oneimage, re_img_for_mark = Predict(t_img.correct_trim).re_img
				len_images = len(re_num_index_oneimage)
				if READ_NUMBERS - len_images < 0:
					times = (READ_NUMBERS - len_images) * -1
					for time in range(times):
						re_num_index_oneimage.pop((len_images - 1) - time)
						error_dict[org_name].append("ERROR:文字数超過")
				suspicious_nums = []
				for index, (file_index, oneimage28) in enumerate(re_num_index_oneimage.items()):
					pre_num = predict_nnc(oneimage28)
					if pre_num is None:
						pre_num = [["" for i in range(READ_NUMBERS)], ""]
						error_dict[org_name].append("ERROR:空白あり")
					elif pre_num[0][0][pre_num[1]] < 0.95:
						suspicious_nums.append(index)
					for per in pre_num[0][0]:
						inspection_dict[f"{org_name}_{file_index}"].append(round(per * 100, 2))
					inspection_dict[f"{org_name}_{file_index}"].append(pre_num[1])
					inspection_dict[f"{org_name}_{file_index}"].append(oneimage28)
					all_dict[org_name].append(str(pre_num[1]))
				all_dict[org_name].append(re_img_for_mark)
				all_dict[org_name].append(suspicious_nums)
			except Exception as e:
				error_log.exception(f"ファイル名：{org_name}、正常に認識できませんでした。\n{traceback.format_exc()}")
				error_dict[org_name].append("ERROR:読取エラー")
				for i in range(READ_NUMBERS + 2):
					all_dict[org_name].append(None)
		df = pd.DataFrame.from_dict(all_dict, orient='index').rename(columns={
			0: 'PIL_image',
			1: 'PATH',
			READ_NUMBERS + 2: "img_for_mark",
			READ_NUMBERS + 3: "sus_nums"
		})
		df["ERROR"] = pd.Series(error_dict)
		if variables.test_flag:
			inspect_df = pd.DataFrame.from_dict(inspection_dict, orient='index').rename(columns={
				0: 'ファイル名',
				10: '認識結果',
				11: 'num_image'
			})
			# inspect_df = inspect_df.drop('num_image')
			inspect_df.to_csv(RESULT, mode='a')
		return df
	except Exception as e:
		sg.popup_error(f"{ERROR_MESS[1]}\n\n{repr(e)}", title=ERROR_MESS[0])
		info_log.info("---ERROR OCCURRED---")
		sys.exit(error_log.exception(f"致命的なエラー\n{traceback.format_exc()}"))
