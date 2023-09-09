import itertools
import zipfile

def un_zip(passwd_string, min_len, max_len, zFile):
    for len in range(min_len, max_len+1):
        to_attempt = itertools.product(passwd_string, repeat = len)
        for attempt in to_attempt:
            passwd = ''.join(attempt)
            print(passwd)
            try:
                zFile.extractall(pwd = passwd.encode())
                print(f"비밀번호는 {passwd} 입니다.")
                return 1
            except:
                pass

passwd_string = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

zFile = zipfile.ZipFile(r'절대경로')

min_len = 1
max_len = 5

unzip_result = un_zip(passwd_string, min_len, max_len, zFile)

if unzip_result == 1:
    print("암호 찾기에 성공하였습니다.")
else:
    print("암호 찾기에 실패하였습니다.")