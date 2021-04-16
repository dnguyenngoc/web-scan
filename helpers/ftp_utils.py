from ftplib import FTP as FTPU
import urllib.request as urllib2
from PIL import Image
import io


class FTP(object):
    def __init__(self, url, username, password):
        self.username = username
        self.password = password
        self.url = url
        self.sess = self.connect()
       
    def connect(self):
        ftp = FTPU()
        ftp.connect(self.url)
        ftp.login(self.username, self.password) 
        return ftp
    
    def list(self, path):
        return self.sess.nlst(path)
    
    def read(self, path):
        fh = urllib2.urlopen('ftp://{user}:{pw}@{host}/{path}'.format(
            user = self.username, pw=self.password, host=self.url, path = path))
        return fh.read()
    
    def move(self, source, destination):
        self.sess.rename(source, destination)
        
    def upload_np_image(self, np_image, path, file_type):
        image = Image.fromarray(np_image.astype('uint8'))
        temp = io.BytesIO()
        image.save(temp, format=file_type)
        temp.seek(0)
        self.sess.storbinary('STOR ./{path}'.format(path=path), temp)

    def chdir(self, dir): 
        try:
            self.sess.mkd(dir)
        except:
            pass


    def close(self):
        if self.sess != None:
            self.sess.quit()
