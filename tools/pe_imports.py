"""打印指定 PE 文件的导入 DLL 列表。用法: python tools/pe_imports.py <file>..."""
import sys

import pefile

for path in sys.argv[1:]:
    pe = pefile.PE(path, fast_load=True)
    pe.parse_data_directories(
        directories=[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_IMPORT']])
    print(path.split("site-packages")[-1])
    for entry in pe.DIRECTORY_ENTRY_IMPORT:
        print("   ", entry.dll.decode())
