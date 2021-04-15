

# database
# SQLALCHEMY_DATABASE_URL = "postgresql://web-scan:1q2w3e4r@192.168.15.18:5432/web-scan"
SQLALCHEMY_DATABASE_URL = "sqlite:///./database/web-scan.db"

# Server type
IDENTITY_CARD = 'identity_card'
DISCHARGE_RECORD = 'discharge_record'

# FTP config
FTP_USERNAME = 'upload'
FTP_PASSWORD = 'raspberry'
FTP_URL = '192.168.15.17'
FTP_PORT = 21
