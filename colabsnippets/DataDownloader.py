import time
import subprocess
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

from .utils import mk_dir_if_not_exists

class DataDownloader():
  def __init__(self, data_dir = './data'):
    auth.authenticate_user()
    gauth = GoogleAuth()
    gauth.credentials = GoogleCredentials.get_application_default()
    self.drive = GoogleDrive(gauth)
    self.data_dir = data_dir

  def download_shard_data(self, db, shard, data_keys):
    data_dir = "./data/{}".format(db)
    for data_key in data_keys:
      zip_file = "{}.7z".format(data_key)
      self.drive.CreateFile({ 'id': shard[data_key] }).GetContentFile("{}/{}".format(data_dir, zip_file))
      ts = time.time()
      subprocess.run(["p7zip", "-d", "./{}".format(zip_file)], cwd = data_dir)
      print("unzipping {} done in {}s".format(zip_file, time.time() - ts))

  def download_data(self, dbs_dict, additional_data_keys = []):
    mk_dir_if_not_exists(self.data_dir)

    ts = time.time()

    for db in dbs_dict.keys():
      print("downloading data for db: {}".format(db))
      mk_dir_if_not_exists("{}/{}".format(self.data_dir, db))

      for idx, shard in enumerate(dbs_dict[db]):
        print("downloading data for shard {}".format(idx))
        self.download_shard_data(db, shard, ['images'] + additional_data_keys)

    print("download_data - total time: {}s".format(time.time() - ts))