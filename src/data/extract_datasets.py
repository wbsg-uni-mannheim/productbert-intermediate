import requests
from zipfile import ZipFile
from pathlib import Path
from tqdm import tqdm

DATASETS = [
    'http://data.dws.informatik.uni-mannheim.de/largescaleproductcorpus/data/v2/repo-download/repo-data.zip'
]


def download_datasets():
    for link in DATASETS:

        '''iterate through all links in DATASETS 
        and download them one by one'''

        # obtain filename by splitting url and getting
        # last string
        file_name = link.split('/')[-1]

        print("Downloading file:%s" % file_name)

        # create response object
        r = requests.get(link, stream=True)

        total_size_in_bytes = int(r.headers.get('content-length', 0))
        block_size = 1024
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(f'../../data/{file_name}', 'wb') as file:
            for data in r.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)

        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print("ERROR, something went wrong")

        print("%s downloaded!\n" % file_name)

    print("All files downloaded!")
    return



def unzip_files():
    for link in DATASETS:
        file_name = link.split('/')[-1]
        # opening the zip file in READ mode
        with ZipFile(f'../../data/{file_name}', 'r') as zip:
            # printing all the contents of the zip file
            zip.printdir()

            # extracting all the files
            print('Extracting all the files now...')
            zip.extractall(path='../../')
            print('Done!')


if __name__ == "__main__":
    Path('../../data/').mkdir(parents=True, exist_ok=True)
    download_datasets()
    unzip_files()
