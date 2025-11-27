# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['launch_flet.py'],
    pathex=[],
    binaries=[],
    datas=[('Model', 'Model'), ('C:\\Users\\Simon\\Documents\\GiHub\\Madakappy\\.venv\\Lib\\site-packages\\rasterio\\gdal_data', 'gdal_data'), ('C:\\Users\\Simon\\Documents\\GiHub\\Madakappy\\.venv\\Lib\\site-packages\\pyproj\\proj_dir\\share\\proj', 'proj_data')],
    hiddenimports=['rasterio.sample', 'rasterio._io', 'rasterio._base', 'rasterio._shim', 'rasterio.crs', 'rasterio.vrt', 'pyogrio', 'fiona', 'pandas._libs.window.aggregations', 'pandas._libs.groupby'],
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
    name='MadaKapPy',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    version='C:\\Users\\Simon\\AppData\\Local\\Temp\\b6e37366-ccb0-4bd0-98a5-f162b6620725',
    icon=['app\\assets\\favicon.ico'],
)
