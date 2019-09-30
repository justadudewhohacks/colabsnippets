drive = None

def init_drive(pydrive, oauth2client, auth):
  auth.authenticate_user()
  gauth = pydrive.auth.GoogleAuth()
  gauth.credentials = oauth2client.client.GoogleCredentials.get_application_default()
  global drive
  drive = pydrive.drive.GoogleDrive(gauth)

def get_global_drive_instance():
  return drive