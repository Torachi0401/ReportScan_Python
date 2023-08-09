# utf-8
import logging

import configparser
from collections import defaultdict
from datetime import datetime

VERSION = "1.0.0"

# GUIで表示するメッセージ系===============================================================================================
START_TITLE = "フォルダ選択"
MEIRYO_FONT = "メイリオ"
READ_FOLDER = "読取フォルダ"
OUTPUT_FOLDER = "出力フォルダ"
GUIDE_MESS = f"{READ_FOLDER}と{OUTPUT_FOLDER}を選択してください。"
DECISION_BTN = "決定"
CANCEL_BTN = "キャンセル"
WAIT_MESS = ["処理中です。しばらくお待ちください。", "入院セット申込書読取"]
WAIT_MESS_TIME = 60
NO_PDF_TITLE = "読取エラー"
NO_PDF_MESS = '指定されたフォルダにPDFファイルがありません。\n' \
              'フォルダが間違っていないかご確認ください。\n' \
              '\n' \
              f'{READ_FOLDER}：%s\n' \
              '\n' \
              '再度選択する場合は「OK」を、中止する場合は「Cancel」を押してください'
SEC_SEARCH_MESS = ["フォルダ検索", "読取先フォルダを指定してください"]
COMP_MESS = ["完了", "読取とファイルリネームが完了しました。"]
ERROR_MESS = ["エラー", f"エラーが発生しました。\n管理者にお問い合わせください。"]
RESTART_MES = ["再実行", "設定を初期値に変更しました。\n設定再読み込みのためシステムを再起動してください。\n終了します。"]
BLANK = ""
NO_CONF_MESS = "CONFIG.INIファイルが見つかりませんでした。"
UNREWRITED_MESS = "CONFIG.INIファイルに書き込めませんでした。"
NO_MOVE_MESS = ["No files", "読取結果を編集した結果、正常なファイルが1つもありませんでしたので、処理終了します。"]
CLOSE_MESS = "選択画面 閉鎖"
REWRITE_MESS = "CONFIG.INIファイル 更新"
HIDE_NEXT_TITLE = "次回から非表示"
# Pysimpleguiで使用するキー
KEY_READ_FOLDER = "key_read_folder"
KEY_OUTPUT_FOLDER = "key_output_folder"
KEY_OK = "key_ok"
KEY_CANCEL = "key_cancel"
KEY_HIDE_NEXT = "key_hide"
KEY_CHANGE_OUTPUT_FOLDER = "key_change_output_folder"
KEY_START_SHOW = "key_start_show"
KEY_START_IMG = "key_start_img"

# その他のメッセージ系
SEC_EXIT_MESS = "読取フォルダの選択がキャンセルされました。"

# 色（GBR）==============================================================================================================
WHITE_COLOR = (255, 255, 255)
BLUE_COLOR = (0, 0, 255)
RED_COLOR = (255, 0, 0)
GREEN_COLOR = (0, 255, 0)
# sRGB
LIGHT_GREEN = "#99FFCC"
LIGHT_BLUE = "#65A3B8"
OCHER = "#57570A"
RED = "red"

# date==================================================================================================================
PAST_DATE = datetime.today().strftime('%Y%m%d')

# Log setting===========================================================================================================
LOG_PATH = "../logs"
LOG_FILENAME = f"{LOG_PATH}/{PAST_DATE}_%s.log"
LOG_FORMAT = "[%(asctime)s]:%(name)s:%(filename)s:%(funcName)s:%(lineno)d:%(levelname)s:%(message)s"
INFO_LOG = "INFO_LOG"
ERROR_LOG = "ERROR_LOG"

# パス類=================================================================================================================
POPPER_BIN = "../libs/poppler/Library/bin"
CONFIG_PATH = "../libs/CONFIG.INI"
NNP_PATH = "../libs/results.nnp"
SUBFORDER_SEARCH = "**/*"
RESULT = fr"{LOG_PATH}/{PAST_DATE}_result.csv"

# 拡張子
EXT_PDF = ".pdf"
EXT_PNG = ".png"

# データ集計するためのdictの設定
all_dict = defaultdict(list)  # GUIで表示させるためにすべての情報を保存させるオブジェクト
"""構造
    Dict{ファイル名のみ(str):
                        List[
                            ⓪PILイメージ(PIL.PngImagePlugin.PngImageFile),
                            ①ファイルのフルパス(str),
                            ②1文字目(str),
                            ③2文字目(str),
                            ④3文字目(str),
                            ⑤4文字目(str),
                            ⑥5文字目(str),
                            ⑦6文字目(str),
                            ⑧7文字目(str),
                            ⑨8文字目(str),
                            ⑩マーキングした画像(np.ndarray)、
                            ⑪認識率が悪い数字(List[int])
                        ]
    }
"""
error_dict = defaultdict(list)  # GUIでエラー情報を表示させるためにエラー情報だけのオブジェクト
inspection_dict = defaultdict(list)  # 認識率のみをCSVとして出力するためのオブジェクト
"""構造
    Dict{ファイル名と数字のインデックス(str):
                        List[
                            ⓪0の確率(float64),
                            ①1の確率(float64),
                            ②2の確率(float64),
                            ③3の確率(float64),
                            ④4の確率(float64),
                            ⑤5の確率(float64),
                            ⑥6の確率(float64),
                            ⑦7の確率(float64),
                            ⑧8の確率(float64),
                            ⑨9の確率(float64),
                            ⑩一番高い確率の数字(int64),
                            ⑫この数字の画像(np.ndarray)
                        ]}
"""


# CONFIG_SECTIONS=======================================================================================================
UTF8 = 'UTF-8'
CONFIG_INI = configparser.ConfigParser(comment_prefixes='/', allow_no_value=True)
CONFIG_INI.read(CONFIG_PATH, UTF8)
SETTINGS = "Settings"
READ_PATH = CONFIG_INI["ReadPath"]["Path"]
OUTPUT_PATH = CONFIG_INI["OutputPath"]["Path"]
READ_AMOUNT = int(CONFIG_INI[SETTINGS]["read_amount"])
DPI = int(CONFIG_INI[SETTINGS]["dpi"])
ALLOWABLE_DIFF = int(CONFIG_INI[SETTINGS]["allowable_diff"])
THRESH_SPIKE = int(CONFIG_INI[SETTINGS]["thresh_spike"])
CHARACTER_SIZE = int(CONFIG_INI[SETTINGS]["character_size"])
BLANK_DISTANCE = int(CONFIG_INI[SETTINGS]["blank_distance"])
HORIZON_THREAD = int(CONFIG_INI[SETTINGS]["horizon_thread"])
TRIM_TOP = int(CONFIG_INI[SETTINGS]["trim_top"])  # トリム：上座標
TRIM_BOTTOM = int(CONFIG_INI[SETTINGS]["trim_bottom"])  # トリム：下座標
TRIM_LEFT = int(CONFIG_INI[SETTINGS]["trim_left"])  # トリム：左座標
TRIM_RIGHT = int(CONFIG_INI[SETTINGS]["trim_right"])  # トリム：右座標
READ_NUMBERS = int(CONFIG_INI[SETTINGS]["read_numbers"])

# 画像ライブラリ=========================================================================================================
NUMPY = "NumPy"
PILLOW = "Pillow"

# test_message==========================================================================================================
BLANK_SPACE = "blank space"
NO_BLANK_SPACE = "no blank space"
TRIMMED_IMAGE = "trimmed image"
MARKED_IMAGE = "marked image"
VERTICAL_ERASE = "vertical_erase"
HORIZONTAL_ERASE = "horizontal_erase"


info_log = logging.getLogger(INFO_LOG)
info_log.setLevel(logging.INFO)
error_log = logging.getLogger(ERROR_LOG)
error_log.setLevel(logging.ERROR)

DEFAULT_SETTING = {
    "read_amount": 100,
    "dpi": 150,
    "allowable_diff": 20,
    "thresh_spike": 50,
    "character_size": 16,
    "blank_distance": 70,
    "horizon_thread": 150,
    "trim_top": 10,
    "trim_bottom": 95,
    "trim_left": 700,
    "trim_right": 1200,
    "read_numbers": 8
}
