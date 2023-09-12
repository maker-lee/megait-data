
회사에서 pip 설치를 막는 경우 
1) 현재 설치된 목록을 텍스트 파일로 저장

pip freeze > 텍스트파일경로



2) 텍스트 파일에 명시된 패키지들을 일괄 설치 

pip install --upgrade -r 텍스트파일경로

(단, 이 방법은 네트워크가 가능한 환경에서만 동작)


3) 외부컴퓨터에서 pip를 통해 패키지들을 특정 폴더에 다운로드받기

pip download -d 폴더경로 -r 텍스트파일경로

ex) pip download -d ./hello -r pypackage.txt

다운로드 후 패키지 목록이 백업된 텍스트 파일을 패키지 폴더안에 함께 보관

---> 이 폴더를 통째로 설치하고자하는 PC에 가져감

설치시에는 패키지와 텍스트파일이 포함된 폴더 위치에서 명령프롬프트를 실행

pip install --no-index --find-links -r 텍스트파일경로

ex) pip install --no-index --find-links -r pypackage.txt