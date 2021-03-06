import os


ENVIRONMENT = os.getenv('ENVIRONMENT')


SMALL_HEIGHT = 720

# DATABASE
SQLALCHEMY_DATABASE_URL = "postgresql://webscan:1q2w3e4r@192.168.15.18:5432/webscan"


# SERVICE TYPE
IDENTITY_CARD = 'identity-card'
DISCHARGE_RECORD = 'discharge-record'


# IDENTITY_CARD SERVICE CONFIG
IDENTITY_CARD_IMPORT_DIR = 'identity-card/import/'
IDENTITY_CARD_EXPORT_DIR = 'identity-card/export/'

# STATUS DOC TYPE
IMPORT_TYPE_NAME = 'import'
EXPORT_TYPE_NAME = 'export'
BAD_TYPE_NAME = 'bad'
TRANSFORM_TYPE_NAME = 'transform'


BE_PORT = 8081
FE_PORT = 8080
FTP_PORT = 21
if ENVIRONMENT == 'staging':
    # FTP CONFIG
    FTP_USERNAME = 'pot'
    FTP_PASSWORD = 'D@123123'
    FTP_URL = '161.117.87.31'
    # BE SERVICE
    BE_HOST = '161.117.87.31'
    # FE SERVICE
    FE_HOST = '161.117.87.31'
elif ENVIRONMENT == 'production':
    # FTP CONFIG
    FTP_USERNAME = 'upload'
    FTP_PASSWORD = 'raspberry'
    FTP_URL = '192.168.15.17'
    # BE SERVICE
    BE_HOST = '192.168.15.19'
    # FE SERVICE
    FE_HOST = '192.168.15.19'
