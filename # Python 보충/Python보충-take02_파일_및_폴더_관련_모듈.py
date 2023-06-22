import sys # 현재 시스템의 정보를 제공하는 모듈
import os # 운영체제의 기능에 접근할 수 있는 모듈


print(sys.platform) # 현재 운영체제 이름 조회

# 경로 문자열 관련 기능 
# -> './' 혹은 '.' 는 현재 폴더라는 의미
# '../'는 상위 폴더라는 의미 

# 현재 폴더 내의 하위 항목들의 이름을 리스트로 리턴 받음
ls = os.listdir('./')
print(ls)

# 특정 폴더나 파일이 존재하는지 확인 -> 상대결로일 경우 현재 소스파일 기준 
k = os.path.exists('./hello') # os 모듈, 안의 path, 안에 있는지 없는지

# 이 위치에 대한 절대 경로 리턴, 존재하지 않더라도 경로값은 확인 가능ㅇ 
print(os.path.abspath('./hello'))


# 폴더의 생성과 삭제
if os.path.exists('./hello') == False : # 폴더가 없다면
    # 없으면 생성
    os.mkdir('./hello')
else :
    # 있으면 삭제 (빈 폴더만 삭제 가능)
    os.rmdir('./hello')
    

# 파일이나 폴더 검색 

import glob as gl # 파일 이름으로 패턴 검색할 때 사용

# 현재 폴더에 있는 모든 하위 요소들 조회
ls = gl.glob('*')

# 현재 폴더에서 '.ipynb'로 끝나는 모든 요소들 조회
ls = gl.glob("*.ipynb")
# 현재 폴더에서 2를 포함하는 파일 조회 
ls = gl.glob("*2*") 


# 폴더 트리 생성 및 삭제
import shutil # 파일 폴더 관련 작업에 확장된 기능을 제공

# 현재 작업 위치에서 python이라는 이름의 폴더가 없다면? 
if os.path.exists('python') == False :

    # 순환적으로 폴더를 생성함 -> exist_ok = True 옵션은 이미 존재하더라도 에러 발생 안함
    os.makedirs('python/test/hello/word',exist_ok=True) # 기존에 있으면 그냥 넘어가고, 없으면 만들어짐
else :
    # 비어있지 않은 폴더도 강제 삭제 -> 존재하지 않는 폴더인 경우 에러
    shutil.rmtree('python')
    print('삭제')

# 폴더의 이동과 폴더 트리 복사
# v폴더의 이동
if os.path.exists('python') :
    # 이름을 바꿔서 이동한다.
    shutil.move('python','../created') 
    # created 폴더를 생성해서 이름을 바꿔서 이동시킨다.

#폴더의 복사
if os.path.exists('created') :
    shutil.copytree('../created','./copy')
    print('python ')




# 파일의 복사 및 삭제 작업
# os 모듈 내의 path 객체가 갖는 exsists()함수를 사용
# -> 'hello.txt'라는 파일이 존재하지 않는다면?

if os.path.exists('hello.txt') == False :
    # 테스트용 파일 생성
    with open('hello.txt','w',encoding="utf-8") as f :
        f.write('i told you')
    # 생성한 파일을 복사함 -> 이미 존재할 경우 덮어씀 
    shutil.copy('hello.txt','world.txt')
# 그렇지 않다면? hello.txt 파일이 존재한다면???
else : # 삭제한다.
    os.remove('hello.txt')
    os.remove('world.txt')





