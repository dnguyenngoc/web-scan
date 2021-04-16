SMALL_HEIGHT = 720

# DATABASE
# SQLALCHEMY_DATABASE_URL = "sqlite:///./database/web-scan.db"
SQLALCHEMY_DATABASE_URL = "postgresql://webscan:1q2w3e4r@192.168.15.18:5432/webscan"


# SERVICE TYPE
IDENTITY_CARD = 'identity_card'
DISCHARGE_RECORD = 'discharge_record'


# IDENTITY_CARD SERVICE CONFIG
IDENTITY_CARD_IMPORT_DIR = 'identity_card/import/'
IDENTITY_CARD_EXPORT_DIR = 'identity_card/export/'


# FTP CONFIG
FTP_USERNAME = 'upload'
FTP_PASSWORD = 'raspberry'
FTP_URL = '10.1.33.76'
FTP_PORT = 21
