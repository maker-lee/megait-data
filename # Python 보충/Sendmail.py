# 메일 보내는 모듈 만드는 중 

import os # 경로 정보를 취득하기 위한 모듈
from smtplib import SMTP # 발송 서버와 연동하기 위한 모듈
from email.mime.text import MIMEText # 본문 구성 기능 
from email.mime.application import MIMEApplication # 첨부파일을 MIMEApplication로 변경
from email.mime.multipart import MIMEMultipart # 첨부파일을 본문에 추가한다. 

# 보낼때마다 바뀌는 정보를 파라미터로 준다 첨부파일은 없을 수 있으니까 디폴트를 빈 리스트로 준다. 
def sendMail(from_addr,to_addr, subject,content,files=[]) :
    # 컨텐츠 형식 plain = 텍스트 / html과 plain이 가능함 
    content_type = "plain"

    # 로그인 계정이름 (네이버는 아이디만, 구글은 메일 주소 통)
    username = 'leejisun0529@gmail.com'

    # 비밀번호 (네이버는 개인 비밀번호, 2단계인증시 애플리케이션 비밀번호, 구글 앱비밀번호)
    password = 'umoecunarjrrbxoe'

    # 구글 발송 서버 주소와 포트(고정값)
    smtp = "smtp.gmail.com"
    port = 587

    # # 네이버 발송 서버 주소와 포트 (고정값)
    # smtp = "smtp.naver.com"
    # port = 465

    # 메일 발송 정보를 저장하기 위한 객체
    msg = MIMEMultipart()

    msg['Subject'] = subject # 메일제목
    msg['From'] = from_addr # 보내는 사람
    msg['To'] = to_addr  # 받는 사람

    # 본문 설정 -> 메일의 내용과 형식 지정
    msg.attach(MIMEText(content,content_type))

    # 리스트 변수의 원소가 하나라도 존재할 경우 True
    if files :
        for f in files :
            # 바이너리 형식으로 읽는다.
            with open(f,'rb') as a_file :
                # 전체 경로에서 파이르이 이름만 추출한다
                basename = os.path.basename(f)
                # 파일의 내용과 파일 이름을 메일에 첨부할 형식으로 변환
                part = MIMEApplication(a_file.read(),Name=basename)

                # 파일 첨부 
                part['Content-Disposition'] = 'attachment; filename="%s"' % basename
                msg.attach(part)


    # 메일 보내기 
    mail = SMTP(smtp) 

    # 메일 서버 접속
    mail.ehlo()
    # 메일 서버 연동 설정
    mail.starttls()
    # 메일 서버 로그인
    mail.login(username, password)
    # 메일 보내기
    mail.sendmail(from_addr, to_addr, msg.as_string())
    # 메일 서버 종료
    mail.quit()


# 현재 파일에서 실행하기 위해서 __name__ 으로 확인한다. (현 파일에서 만든 함수라면 __main__호출 그게 아니라면 임포트하는 파일 이름 호출)

# 모듈 테스트용으로 아래와 같이 함
if __name__ == "__main__" :
    sendMail('leejisun0529@gmail.com',"hare0529@naver.com",'여기는 제목이다','나는 내용이다')