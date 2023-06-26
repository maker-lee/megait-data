# ### 예제 4-1 상하좌우 
# 여행가 A는 N * N 크기의 정사각형 공간 위에 서 있다. 이 공간은 1 * 1 크기의 정사각형으로 나누어져 있다. 가장 왼쪽 위 자료는 (1,1) 이며, 가장 오른쪽 아래 좌표는 (N,N)에 해당한다. 여행가 A는 상,하,좌,우 방향으로 이동할 수 있으며 시작 좌표는 항상 (1,1)이다. 
# 계획서에는 하나의 줄에 띄어쓰기를 기준으로 하여 L,R,U,D 중 하나의 문자가 반복적으로 적혀있다. 각 문자의 의미는 다음과 같다. 

# L 왼쪽으로 한 칸 이동
# R 오른쪽으로 한 칸 이동
# U 위로 한 칸 이동
# D 아래로 한 칸 이동

# 이때 여행가 A가 N * N 크기의 정사각형 공간을 벗어나는 움직임은 무시된다. 
# 예를 들어 (1,1)의 위치에서 L혹은 U를 만나면 무시된다.
# 다음은 N = 5 인 지도와 계획서이다. 

# 계획서와 지도 
# R-R-R-U-D-D- -> (1,1)(1,2)(1,3)(1,4) -> 공간밖은 무시(U)
# 이 경우 6개의 명령에 따라서 여행가가 움직이게 되는 위치는 순서대로 (1,1) r (1,2) r (1,3) r (1,4) u (1,4) d (2,4) d (3,4) 이므로 
# 최종적으로 여행가 A가 도착하게 되는 곳의 좌표는 (3,4)이다. 다시 말해 3행 4열의 위치에 해당하므로 (3,4)라고 적는다.
# 계획서가 주어졌을 때 여행가 A가 최종적으로 도착할 지점의 좌표를 출력하는 프로그램을 작성하시오.

# 입력조건
# 첫째줄에 공간의 크기를 나타내는 N이 주어진다(1~100)
# 둘째줄에 여행가 A가 이동할 계획서 내용이 주어진다. (1~100)

# 출력조건
# X,Y를 공백으로 구분하여 출력한다.

# 예시
# 5
# R R R U D D
# 출력 예시 3 4


# setting
move_x, move_y = [1,1] # 시작지점 
move_rule = {'L':[0,-1],'R':[0,1],'U':[-1,0],"D":[1,0]} # 이동 방법

# input
# n = int(input())
# move = input().split()
n = 5  # 지도의 최대치 
move = ['R','R','R','U','D','D'] # 이동할 커맨드 

for i in move :
    x,y = move_rule[i] # 이동커맨드를 키로 이동할 방향을 값을 받아온다. 
    move_x += x
    move_y += y  
    if 0 == move_x or move_x > n or 0 == move_y or move_y > n: # 범위를 벗어나면 아까 이동한거 실행취소 
        move_x -= x 
        move_y -= y
print(move_x,move_y)
    







# ### 예제 4-2 시각
# 정수 N이 입력되면 00시 00분 00초부터 N시 59분 59초 까지의 모든 시각 중에서 3이 하나라도 포함되는 모든 경우의 수를 구하는 프로그램을 작성하시오
# 예를 들어 1을 입력 했을때 00시00분03초, 00시 13분 30초는 3시 하나라도 포함되어 있으므로 세어야 하는 시각이다.
# 반면에 다음은 3시 하나도 포함되어 있지 않으므로 세면 안되는 시각이다. 00시 02분 55초, 01시 27분 45초

# 입력조건
# 첫째줄에 정수 N이 입력된다 (0~23)
# 출력조건 00시00분00초부터 N시59분59초까지의 모든 시각 중에서 3이 하나라도 포함되는 모든 경우의 수를 출력한다.

# 예시
# 5
# 출력예시 
# 11475


hour = 5 
cnt = 0 

# 모든 시간을 000000 형태로 바꿔서 count 해본다. 
for h in range(hour + 1) : #  N시59분59초로 N시가 포함되므로 1을 더함 
    for m in range(60) :
        for c in range(60) :
            # clock = (str(h)+str(m)+str(c))
            # cnt += clock.count('3')       # 왜 정답과 다르지???  
            if '3' in str(h)+str(m)+str(c) :
                cnt += 1 

print('3이 나오는 횟수는 총 %d개' % (cnt)) # 11475  총 15120개 ??? 








# ### 실전문제 2 왕실의 나이트

# 체스판 (8*8 좌표평면) 특정한 한 칸에 나이트가 서 있다. 나이트는 L 자 형태로만 이동할 수 있으며 정원 밖으로는 나갈 수 없다.
# 나이트는 특정한 위치에서 2가지 경우로 이동할 수 있다.
# (1) 수평으로 2칸 이동한 뒤에 수직으로 1칸 이동하기
# (2) 수직으로 2칸 이동한 뒤에 수평으로 1칸 이동하기 

# 나이트 위치가 주어졌을 때 나이트가 이동할 수 있는 경우의 수를 출력하는 프로그램을 작성하시오
# 이때 행은 1~8, 열은 A~H로 표현된다.

# 만약 나이트가 A1에 있을 때 이동할 수 있는 경우의 수는 다음 2가지다. 
# 오른쪽으로 두칸 이동 후 아래로 한 칸 이동하기(c2)
# 아래로 두칸 이동 후 오른쪽으로 한 칸 이동하기(b3)

# > 입력조건 
# 나이트의 위치 = a1 
# > 출력조건
# 나이트가 이동할 수 있는 경우의 수 = 2
# (c2에서 이동할 수 있는 경우의 수는 6가지)






# 세팅
move = [(-2,1),(-2,-1),(2,-1),(2,1),(1,2),(1,-2),(-1,2),(-1,-2)] # 움직일 수 있는 모든 경우의 수
a_h= ['a','b','c','d','e','f','g','h'] #열 숫자 변환 
max = 8 # 상하좌우 갈 수 있는 최대 위치 

# 위치 받아서 (row,column)으로 정렬
loc= input() 
row = int(loc[1]) 
column = a_h.index(loc[0])+1   #a의 index가 0이라서 느낌있게 1로 맞춰줌 

cnt = 0 
for i in move : 
    if 0 < row + i[0] < max+1 and  0 < column + i[1] < max+1 : # 가로 1~8, 세로 1~8 사이만 카운트 
        # print('이동가능.',row + i[0], column + i[1])
        cnt += 1
print(cnt)







# ### 실전문제 3 게임 개발
# N * M 크기의 직사각형으로 캐릭터는 동서남북 중 한 곳을 바라본다. map의 각 칸은 (A,B)로 나타낼 수 있고, A는 북쪽으로부터 떨어진 칸의 개수, B는 서쪽으로부터 떨어진 칸의 개수이다. 캐릭터는 상하좌우로 움직일 수 있고 바다로 되어 있는 공간에는 갈 수 없다. 
# (1) 캐릭터는 현재 위치에서 현재 방향을 기준으로 왼쪽방향(반시계 방향으로 90도 회전한 방향)부터 차례대로 갈 곳을 정한다
# (2) 캐릭터의 바로 왼쪽 방향에 아직 가보지 않은 칸이 존재한다면, 왼쪽 방향으로 회전한 다음 왼쪽으로 한 칸을 전진한다. 왼쪽 방향에 가보지 않은 칸이 없다면, 왼쪽 방향으로 회전만 수행하고 1단계로 돌아간다.
# (3) 만약 네 방향 모두 이미 가본 칸이거나 바다로 되어있는 칸인 경우에는 바라보는 방향을 유지한 채로 한 칸 뒤로 가고 1단계로 돌아간다. 단, 이때 뒤쪽 방향이 바다인 칸이라 뒤로 갈 수 없는 경우에는 움직임을 멈춘다.

# 매뉴얼에 따라 캐릭터를 이동시킨 뒤에, 캐릭터가 방문한 칸의 수를 출력하는 프로그램을 만드시오.

# 입력조건
# 첫째 줄에 맴의 세로 크기 N과 가로 크기 M을 공백으로 구분하여 입력한다. (3~50)
# 둘째 줄에 게임 캐릭터가 있는 칸의 좌표 (a,b) 와 바라보는 방향 d가 각각 서로 공백으로 구분하여 주어진다. 방향 d의 값으로는 다음과 같이 4가지가 존재한다.
# 0 북 1 동 2 남 3서쪽
# 셋째 줄부터 맵이 육지인지 바다인지에 대한 정보가 주어진다. N개의 줄에 맴의 상태가 북쪽부터 남쪽 순서대로 각 줄의 데이터는 서쪽부터 동쪽 순서대로 주어진다. 맵의 외곽은 항상 바다로 되어 있다.
# 0 육지 1 바다
# 처음에 캐릭터는 육지에 위치한다.

# 출력조건
# 첫째줄에 이동을 마친 후 캐릭터가 방문한 칸의 수를 출력한다.

