import pyautogui
import time
import pyperclip


# 네이버에서 자동으로 서울 날씨 검색하는 코드 만들기 

날씨 = ["서울 날씨","시흥 날씨","청주 날씨","부산 날씨","강원도 날씨"]

# 왼쪽 창의 주소 검색창 좌표
addr_x = 197
addr_y = 76
# 날씨 창의 왼쪽 상단 좌표
start_x = 39
start_y = 316
# 날씨 창의 우측 하단 좌표
end_x = 876
end_y = 830

while True :
    time.sleep(3)
    print(pyautogui.position()) # 마우스 좌표 출력
    time.sleep(10)
    break
 

for 지역날씨 in 날씨:
    pyautogui.moveTo(addr_x,addr_y,1)       # 주소 검색창의 좌표로 1초에 걸쳐 이동
    time.sleep(0.2)
    pyautogui.click()
    time.sleep(0.2)
    pyautogui.write("www.naver.com",interval=0.1)   # 네이버 주소를 0.1초 간격으로 영문 입력
    pyautogui.write(["enter"])
    time.sleep(1)

    pyperclip.copy(지역날씨)
    pyautogui.hotkey("ctrl", "v")
    time.sleep(0.5)
    pyautogui.write(["enter"])
    time.sleep(1)
    저장경로 = 지역날씨 + '.png'
    pyautogui.screenshot(저장경로, region=(start_x, start_y, end_x-start_x, end_y-start_y))