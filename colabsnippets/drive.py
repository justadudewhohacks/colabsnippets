drive = None

def init_drive(GoogleDrive, GoogleAuth, GoogleCredentials, auth):
  auth.authenticate_user()
  gauth = GoogleAuth()
  gauth.credentials = GoogleCredentials.get_application_default()
  global drive
  drive = GoogleDrive(gauth)

def get_global_drive_instance():
  return drive