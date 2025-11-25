# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['launch_flet.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=['rasterio._base', 'rasterio._io', 'rasterio._env', 'rasterio._err', 'rasterio._features', 'rasterio._warp', 'rasterio._fill', 'rasterio._features', 'rasterio._path', 'rasterio._sieve', 'rasterio._rasterio', 'rasterio._transform', 'rasterio.sample', 'rasterio.vrt'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='Madakappy',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    version='C:\\Users\\Simon\\AppData\\Local\\Temp\\2fc98899-a6f5-4957-9104-cb0d2cec37d3',
    icon=['app\\assets\\favicon.png'],
)
