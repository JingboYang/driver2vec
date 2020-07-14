import os
import sys
sys.path.append('..')

from multiprocessing import Pool
import time

from constants import *
from utils import GCStorage, GCOpen


storage = GCStorage(PROJECT_ID, GC_BUCKET, CREDENTIAL_PATH)

file_list = storage.list_files('driver_data')[1]
# print(file_list)

def download_file(args):
    folder, fname = args
    
    print(folder, fname)
    local_path = TEMP_FOLDER / 'driver_data' / folder / fname
    cloud_path = Path('driver_data') / folder / fname
        
    count = 0
    while count < 5:
        try:
            storage.download(local_path, cloud_path)
            count = 10
        except:
            time.sleep(1)
            count += 1
            if count == 5:
                print(f'Could not download {args}')

arguments = []
for folder in ['area_2_divided', 'divided_by_area', 'coord_splitted', 'divided_splitted']:
    os.makedirs(str(TEMP_FOLDER / 'driver_data' / folder), exist_ok=True)
    for fname in file_list[folder]:
        arguments.append((folder, fname))


print(len(arguments))

pl = Pool(26)
pl.map(download_file, arguments)