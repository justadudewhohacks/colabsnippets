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

  def download_images_and_landmarks(self, db, images_id, landmarks_id = None):
    data_dir = "./data/{}".format(db)
    self.drive.CreateFile({ 'id': images_id }).GetContentFile("{}/images.7z".format(data_dir))

    ts = time.time()
    subprocess.run(["p7zip", "-d", "./images.7z"], cwd = data_dir)
    print("unzipping images done in {}s".format(time.time() - ts))

    if landmarks_id is not None:
      self.drive.CreateFile({ 'id': landmarks_id }).GetContentFile("{}/landmarks.7z".format(data_dir))
      ts = time.time()
      subprocess.run(["p7zip", "-d", "./landmarks.7z"], cwd = data_dir)
      print("unzipping landmarks done in {}s".format(time.time() - ts))

  def download_data(self, dbs_dict):
    mk_dir_if_not_exists(self.data_dir)

    ts = time.time()

    for db in dbs_dict.keys():
      print("downloading data for db: {}".format(db))
      mk_dir_if_not_exists("{}/{}".format(self.data_dir, db))

      for shard in dbs_dict[db]:
        print("downloading data for shard {}".format(shard['shard_id'] if 'shard_id' in shard else '0'))
        self.download_images_and_landmarks(db, shard['images'], shard['landmarks'] if 'landmarks' in shard else None)

    print("download_data - total time: {}s".format(time.time() - ts))