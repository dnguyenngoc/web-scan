import ftplib


class FTP:
    def __init__(self, url, username, password):
        self.username = username
        self.password = password
        self.url = url
        self.sess = ftplib.FTP(url,self.username, self.password)
    
    def load_file(self, path):
        file = open(path,'rb')     
        res = self.sess.retrlines('RETR ' + path, self.sess.write)
        file.close() 
        return res
        
    def create_file(self, path):
        file = open(path,'wb')     
        self.sess.storbinary('STOR ' + path, file)
        file.close()  
    
    def close(self)
        if self.sess != None:
            self.sess.quit()