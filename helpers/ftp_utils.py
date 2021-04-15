import ftplib


class FTP:
    def __init__(self, url, username, password, port):
        self.username = username
        self.password = password
        self.url = url
        self.port = port
        self.sess = self.connect()
        
    def connect(self):
        self.sess = ftplib.FTP(self.url, self.username, self.password).login()
    
    def load_file(self, path):
        file = open(path,'rb')     
        res = self.sess.retrlines('RETR ' + path, self.sess.write)
        file.close() 
        return res
        
    def create_file(self, path, file):
        self.sess.storbinary('STOR ' + path, file)
        file.close()  
    
    def close(self):
        if self.sess != None:
            self.sess.quit()
