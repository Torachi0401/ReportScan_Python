# -*- mode: python ; coding: utf-8 -*-


block_cipher = None

def get_nnabla_path():
    import nnabla
    nnabla_path = nnabla.__path__[0]
    return nnabla_path

a = Analysis(['main.py'],
             pathex=[],
             binaries=[('./libs/poppler', './binary')],
             datas=[],
             hiddenimports=[],
             hookspath=[],
             hooksconfig={},
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
dict_tree = Tree(get_nnabla_path(), prefix='nnabla', excludes=["*.py*"])
a.datas += dict_tree

pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

exe = EXE(pyz,
          a.scripts, 
          [],
          exclude_binaries=True,
          name='OCR申込書読取',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True,
          disable_windowed_traceback=False,
          target_arch=None,
          codesign_identity=None,
          entitlements_file=None , version='..\\pyinstaller\\VersionInfo.txt', icon='..\\pyinstaller\\B03.ico')
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas, 
               strip=False,
               upx=True,
               upx_exclude=[],
               name='OCR申込書読取')
