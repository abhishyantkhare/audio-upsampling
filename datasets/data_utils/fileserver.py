import json
import os
import shutil

from ftplib import FTP

TEMP_DIR = 'FILESERVER_TEMP'
SERVER_CONFIG = 'datasets/server_config'

with open(SERVER_CONFIG) as f:
    server_config_dict = json.loads(f.read())

FILESERVER_URL = server_config_dict['FILESERVER_URL']
FILESERVER_DIR = server_config_dict['FILESERVER_DIR']
FILESERVER_USER = server_config_dict['FILESERVER_USER']
FILESERVER_PASSWORD = server_config_dict['FILESERVER_PASSWORD']


class Fileserver:

    def __init__(self, dataset='fma_small'):
        self.ftp = FTP(FILESERVER_URL)
        self.ftp.login(FILESERVER_USER, FILESERVER_PASSWORD)
        self.ftp.cwd(FILESERVER_DIR + '/' + dataset)
        self.pwd = FILESERVER_DIR + '/' + dataset
        if os.path.isdir(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)
        os.mkdir(TEMP_DIR)

    def __del__(self):
        self.ftp.close()

    def get_local_file(self, filename):
        if not os.path.isdir(TEMP_DIR + '/' + os.path.dirname(filename)):
            os.makedirs(TEMP_DIR + '/' + os.path.dirname(filename))
        return TEMP_DIR + '/' + filename

    def clear_cache(self):
        if os.path.isdir(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)
        os.mkdir(TEMP_DIR)

    def ls(self, directory='.', relative=True):
        # List directories
        filenames = self.ftp.nlst(directory)
        # Remove PWD from filenames if necessary
        if relative:
            filenames = [filename[len(self.pwd)+1:] for filename in filenames]
        return filenames

    def cd(self, directory):
        self.ftp.cwd(directory)

    def mkdir(self, directory):
        self.ftp.mkd(directory)

    def download(self, filenames):
        if type(filenames) == str:
            filenames = [filenames]
        for filename in filenames:
            if not os.path.isdir(TEMP_DIR + '/' + os.path.dirname(filename)):
                os.makedirs(TEMP_DIR + '/' + os.path.dirname(filename))
            print('Downloading', filename)
            with open(TEMP_DIR + '/' + filename, 'wb') as f:
                self.ftp.retrbinary('RETR ' + './' + filename, f.write)

    def upload(self, filenames):
        if type(filenames) == str:
            filenames = [filenames]
        for filename in filenames:
            print('Uploading', filename)
            with open(TEMP_DIR + '/' + filename, 'rb') as f:
                self.ftp.storbinary('STOR ' + './' + filename, f)
