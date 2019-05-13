from ftplib import FTP

ftp = FTP('fileserver.brianlevis.com')
ftp.login('cs182', 'donkeyballs')
ftp.cwd('/BRIANDISK/tensorpros/pytorch')
pwd = ftp.pwd()

for filename in ['xr_model.ckpt']:
    with open(filename, 'rb') as fp:
        ftp.storbinary('STOR ' + './' + filename, fp)

ftp.close()

