import os
import zipfile
from LCover import logger
from LCover.utils.common import split_images, get_size
from LCover.entity.config_entity import (DataIngestionConfig)
import time
import shutil

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    
    def download_file(self)-> str:
        '''
        Fetch data from the url
        '''

        try: 
            pass

        except Exception as e:
            raise e
        

    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        pass

    def prepare_data(self):
        logger.info("Splitting the raw image")
        tic = time.time()
        split_images(DATA_DIR=self.config.unzip_dir, OUTPUT_DIR=self.config.root_dir, TARGET_SIZE = self.config.image_size)
        mins,sec = divmod(time.time()-tic,60)
        print(f"Execution completed in {mins} minutes and {sec:.2f} seconds.")
        shutil.copy(os.path.join(self.config.unzip_dir,'train.txt'), os.path.join(self.config.root_dir,'train.txt'))
        shutil.copy(os.path.join(self.config.unzip_dir,'val.txt'), os.path.join(self.config.root_dir,'val.txt'))
        shutil.copy(os.path.join(self.config.unzip_dir,'test.txt'), os.path.join(self.config.root_dir,'test.txt'))
