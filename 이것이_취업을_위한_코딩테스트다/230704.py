# '''
# 07 럭키 스트레이트 P321 구현
# 캐릭터의 점수를 N이라고 할 때, 자릿수를 기준으로 점수 N을 반으로 나누어 왼쪽 부분의 각 자릿수의 합과 오른쪽 부분의 각 자릿수의 합을 더한 값이 동일한 상황을 의미합니다. 
# 현재 점수가 123402라면 왼쪽 부분의 각 자릿수의 합은 1+2+3, 오른쪽 부분의 각 자릿수의 합은 4+0+2이므로 두 합이 6으로 동일하여 럭키 스트레이르를 사용할 수 있다.

# 현재 점수 N이 주어졌을 때 럭키스트레이트를 쓸 수 있는 상태인가요?

# 입력조건
# 첫째줄에 점수 N이 정수로 주어짐(자릿수는 항상 짝수)

# 출력조건
# 럭키 스트레이트를 사용할 있다면 LUCKY를 , 없으면 READY를 반환

# 예시 1 123402 = LUCKY
# 예시 2 7755 = READY

# '''

# '''문제 해결 프로세스
# 입력받은 문자를 절반으로 자른다
# 각각 변수에 넣는다. 
# 숫자로 변환한다. 
# 리스트로 만들어서 합계를 구한다. 
# 두 개가 같으면 LUCKY, 아니면 READY 
# '''


N = '123402'
first,second = N[:len(N)//2], N[len(N)//2:] 
if sum(list(map(int,first))) == sum(list(map(int,second))) : 
    print('LUCKY') 
else :
    print("READY")



# '''
# 08. 문자열 재정렬

# 알파벳 대문자와 숫자 0~9로 구성된 문자열이 주어졌을 때 모든 알파벳을 오름차순으로 정렬하여 이어서 출력한 뒤, 그 뒤에 모든 숫자를 더한 값을 이어서 출력한다.

# 입력예시 
# K1KA5CB7 \ ABCKK13
# AJKDLSI412K4JSJ9D = ADDIJJKKLSS20

# ord로 정리하면 되지 않을까
# print(ord('1')) # A= 65 1=49 ord를 이용하는 방법......
# '''




# '''문제 해결 프로세스

# 단어를 리스트로 바꾼다
# 크기순으로 정렬한다
# 단어를 숫자로 바꿔본다. 
# 에러나면 문자니까 문자를 더함 
# 에러 안나면 숫자니까 sum 
# 확인이 끝나면 두갤 합친다. 
# '''


stringNum = 'K1KA5CB7'
stringNum = list(stringNum)
stringNum.sort() 

num = 0
string = ''

for i in stringNum :
    try : 
        num += int(i) 
    except :
        string += i

print(string + str(num))


# '''
# 09문자열 압축
# 데이터 처리 전문가가 되고 싶은 "어피치"는 문자열을 압축하는 방법에 대해 공부를 하고 있습니다. 최근에 대량의 데이터 처리를 위한 간단한 비손실 압축 방법에 대해 공부를 하고 있는데, 문자열에서 같은 값이 연속해서 나타나는 것을 그 문자의 개수와 반복되는 값으로 표현하여 더 짧은 문자열로 줄여서 표현하는 알고리즘을 공부하고 있습니다.

# # 간단한 예로 "aabbaccc"의 경우 "2a2ba3c"(문자가 반복되지 않아 한번만 나타난 경우 1은 생략함)와 같이 표현할 수 있는데, 이러한 방식은 반복되는 문자가 적은 경우 압축률이 낮다는 단점이 있습니다. 예를 들면, "abcabcdede"와 같은 문자열은 전혀 압축되지 않습니다. "어피치"는 이러한 단점을 해결하기 위해 문자열을 1개 이상의 단위로 잘라서 압축하여 더 짧은 문자열로 표현할 수 있는지 방법을 찾아보려고 합니다.

# 예를 들어, "ababcdcdababcdcd"의 경우 문자를 1개 단위로 자르면 전혀 압축되지 않지만, 2개 단위로 잘라서 압축한다면 "2ab2cd2ab2cd"로 표현할 수 있습니다. 다른 방법으로 8개 단위로 잘라서 압축한다면 "2ababcdcd"로 표현할 수 있으며, 이때가 가장 짧게 압축하여 표현할 수 있는 방법입니다.

# 다른 예로, "abcabcdede"와 같은 경우, 문자를 2개 단위로 잘라서 압축하면 "abcabc2de"가 되지만, 3개 단위로 자른다면 "2abcdede"가 되어 3개 단위가 가장 짧은 압축 방법이 됩니다. 이때 3개 단위로 자르고 마지막에 남는 문자열은 그대로 붙여주면 됩니다.

# 압축할 문자열 s가 매개변수로 주어질 때, 위에 설명한 방법으로 1개 이상 단위로 문자열을 잘라 압축하여 표현한 문자열 중 가장 짧은 것의 길이를 return 하도록 solution 함수를 완성해주세요.

# 제한 사항
# s의 길이는 1 이상 1,000 이하입니다.
# s는 알파벳 소문자로만 이루어져 있습니다.

# 예시 / 값 
# "aabbaccc"	7
# "ababcdcdababcdcd"	9
# "abcabcdede"	8
# "abcabcabcabcdededededede"	14
# "xababcdcdababcdcd"	17

# 입출력 예에 대한 설명
# 입출력 예 #1
# 문자열을 1개 단위로 잘라 압축했을 때 가장 짧습니다.
# 입출력 예 #2
# 문자열을 8개 단위로 잘라 압축했을 때 가장 짧습니다.
# 입출력 예 #3
# 문자열을 3개 단위로 잘라 압축했을 때 가장 짧습니다.
# 입출력 예 #4
# 문자열을 2개 단위로 자르면 "abcabcabcabc6de" 가 됩니다.
# 문자열을 3개 단위로 자르면 "4abcdededededede" 가 됩니다.
# 문자열을 4개 단위로 자르면 "abcabcabcabc3dede" 가 됩니다.
# 문자열을 6개 단위로 자를 경우 "2abcabc2dedede"가 되며, 이때의 길이가 14로 가장 짧습니다.
# 입출력 예 #5
# 문자열은 제일 앞부터 정해진 길이만큼 잘라야 합니다.
# 따라서 주어진 문자열을 x / ababcdcd / ababcdcd 로 자르는 것은 불가능 합니다.
# 이 경우 어떻게 문자열을 잘라도 압축되지 않으므로 가장 짧은 길이는 17이 됩니다.
# '''

# '''
# 문제 해결 프로세스
'''
앞단어와 뒷단어가 같은지 확인한다.
만약 같으면 숫자단어 형태로 기억한다
만약 같았는데 다르면 숫자는 1이 되고 지금까지 만든 숫자단어는 저장한다
만약 같은적도 없다면 단어를 저장한다.

위를 반복한다. 최대 반복이 문자 길이의 1/2이므로 그만큼 반복하고
문자 슬라이싱은 1개->2개->3개 묶음이 되도록 한다. 

이렇게 만든 단어들을 모두 모아서 길이를 비교해 가장 작은걸 낸다.

'''

s = 'xababcdcdababcdcd'

total = []
origin = [len(s)]
for n in range(1,len(s)//2+1) : # 1개 단어 ~ 전체 단어의 1/2개씩 잘라서 비교할 예정 
    cnt = 1
    word = ''
    word_list = []
    for i in range(0,len(s),n) : # 단어 슬라이싱         
        current_word = s[i:i+n] # 앞 단어
        #print('앞',current_word)
        next_word = s[i+n:i+n+n] # 뒷 단어
        #print('뒤',next_word)
        if current_word == next_word : # 앞 단어가 뒷 단어와 같으면
            cnt += 1
            word = str(cnt)+current_word
        elif current_word != next_word and cnt > 1 : # 앞 단어가 반복됐다가 뒷 단어와 달라지기 시작 
            word_list.append(word)
            cnt = 1
        else : # 앞 단어와 뒷 단어가 같아본적이 없음 
            word_list.append(current_word) 
    #total.append(''.join(word_list))  # 압축된 단어
    total.append(len(''.join(word_list))) # 압축된 단어 수
print(min(total+origin))





'''자물쇠와 열쇠
2020 카카오 신입 공채 

고고학자인 "튜브"는 고대 유적지에서 보물과 유적이 가득할 것으로 추정되는 비밀의 문을 발견하였습니다. 그런데 문을 열려고 살펴보니 특이한 형태의 자물쇠로 잠겨 있었고 문 앞에는 특이한 형태의 열쇠와 함께 자물쇠를 푸는 방법에 대해 다음과 같이 설명해 주는 종이가 발견되었습니다.

잠겨있는 자물쇠는 격자 한 칸의 크기가 1 x 1인 N x N 크기의 정사각 격자 형태이고 특이한 모양의 열쇠는 M x M 크기인 정사각 격자 형태로 되어 있습니다.

자물쇠에는 홈이 파여 있고 열쇠 또한 홈과 돌기 부분이 있습니다. 열쇠는 회전과 이동이 가능하며 열쇠의 돌기 부분을 자물쇠의 홈 부분에 딱 맞게 채우면 자물쇠가 열리게 되는 구조입니다. 자물쇠 영역을 벗어난 부분에 있는 열쇠의 홈과 돌기는 자물쇠를 여는 데 영향을 주지 않지만(!!!), 자물쇠 영역 내에서는 열쇠의 돌기 부분과 자물쇠의 홈 부분이 정확히 일치해야 하며 열쇠의 돌기와 자물쇠의 돌기가 만나서는 안됩니다.(!!!) 또한 자물쇠의 모든 홈을 채워 비어있는 곳이 없어야 자물쇠를 열 수 있습니다.

열쇠를 나타내는 2차원 배열 key와 자물쇠를 나타내는 2차원 배열 lock이 매개변수로 주어질 때, 열쇠로 자물쇠를 열수 있으면 true를, 열 수 없으면 false를 return 하도록 solution 함수를 완성해주세요.

제한사항
key는 M x M(3 ≤ M ≤ 20, M은 자연수)크기 2차원 배열입니다.
lock은 N x N(3 ≤ N ≤ 20, N은 자연수)크기 2차원 배열입니다.
M은 항상 N 이하입니다.
key와 lock의 원소는 0 또는 1로 이루어져 있습니다.
0은 홈 부분, 1은 돌기 부분을 나타냅니다.

key를 시계 방향으로 90도 회전하고, 오른쪽으로 한 칸, 아래로 한 칸 이동하면 lock의 홈 부분을 정확히 모두 채울 수 있습니다.

-> 돌기+돌기 = 불가
-> lock 파트를 벗어나는건 가능
-> lock 파트의 0은 모두 1이 되어야함 

그러면 덧셈을 해서 lock 부분은 모두 1,1,1 / 1,1,1 / 1,1,1 이 되면 되겠넹?


key	                         
[[0, 0, 0], 
[1, 0, 0], 
[0, 1, 1]]	

lock	 
[[1, 1, 1], 
[1, 1, 0], 
[1, 0, 1]]	

result
true
'''

'''
문제 해결 프로세스

lock의 상하좌우를 key의 배열-1 만큼 키운다, 배열의 빈칸은 0으로 처리한다. 
    if key == 0 and lock ==  1
    90도 돌려서 다시 if문을 돈다.
# '''

# key =[[0, 0, 0], [1, 0, 0], [0, 1, 1]]	
# lock = [[1, 1, 1], [1, 1, 0], [1, 0, 1]]	


# def solution(key, lock):    
   
#     import copy
    
#     answer = False

#     # 90도 돌리는 함수 
#     def degree90(a) :
#         n = len(a) # 리스트의 길이 
#         m = len(a[0]) # 리스트 내의 리스트 길이
#         result = [[0] * n for _ in range(m)]  
#         for i in range(n) : # 리스트의 길이만큼 반복
#             for j in range(m) : # 리스트 내의 리스트 길이만큼 반복
#                 result[j][n-i-1] = a[i][j]  # 리스트[][] = a[][]
#         return result 

#     # lock의 구멍 부분 제외 다 10로 바꿈 #[[10, 10, 10], [10, 10, 0], [10, 0, 10]]
#     for a in range(len(lock)) :
#         for b in range(len(lock[a])) :
#             if lock[a][b] == 1 :
#                 lock[a][b] = 10
#             else :
#                 pass 
#     lock_cnt = sum(lock,[]).count(0)

#     # key의 돌기 부분 제외 다 10으로 바꿈 [[10, 10, 10], [0, 10, 10], [10, 0, 0]]
#     for a in range(len(key)) :
#         for b in range(len(key[a])) :
#             if key[a][b] == 1 :
#                 key[a][b] = 0 
#             else :
#                 key[a][b] = 10


# # 이러면 크기가 같을 때만 true가 나옴
#     for _ in range(4) :
#         key90 = degree90(key)
#         res = copy.deepcopy(lock[:]) # 뭐가 됐든 lock과 크기가 같으면 됨 

#         for list1 in range(len(lock)) :
#             for list2 in range(len(lock[list1])) :
#                 res[list1][list2] = lock[list1][list2] + key90[list1][list2]            

#         if sum(res,[]).count(0) == lock_cnt :
#             answer = True
#             break
#         else :
#             key = degree90(key)
#             continue        

#     return answer

# print(solution(key, lock))


'''
lock에 있는 0을 모두 센다
key와 lock을 더한다.
0을 카운트한다
0이 모두 사라지면 성공

'''
2
1
1
1
#------------------#
2
0
0
2
#------------------#

2
1
2
1

2
1
1
1


lock = [[1, 1, 1], [1, 1, 0], [1, 0, 1]]
key =[[1, 0], [0, 1]]	

# 바위에 구멍이 몇개인지, 키에 돌기가 몇개인지 확인하기 
lock_cnt0 = sum(lock,[]).count(0)
key_cnt0 = sum(key,[]).count(1)

# 모든 문이 열려있거나, 키에 돌기가 없거나 키가 없으면 -> 둘중 하나가 불량이면 
if lock_cnt0 == 0 :
    print('오픈 더 도어~')
if len(key) == 0 or key_cnt0 == 1:
    print('불가능')

# key한테 lock이 오는 방식으로 해보자공 
lock_size = len(lock[0])
key_size = len(key[0])

x = 0
y = 0 

# 90도 돌리는 함수 
def degree90(a) :
    n = len(a) # 리스트의 길이 
    m = len(a[0]) # 리스트 내의 리스트 길이
    result = [[0] * n for _ in range(m)]  
    for i in range(n) : # 리스트의 길이만큼 반복
        for j in range(m) : # 리스트 내의 리스트 길이만큼 반복
            result[j][n-i-1] = a[i][j]  # 리스트[][] = a[][]
    return result 

while True : 
    degree90(key)
    
    for i in range(key_size) : #가로
        for j in range(key_size) : # 세로
            try : 
                print(lock[i+x][j+y] + key[i][j])             
           
                if x <= lock_size : # 바위의 범위보다 작을때        
                    x += 1
                else :
                    x = 0                    
            except :
                # 범위를 벗어나면 한 줄 아래로 내린다. 
                y += 1 
                print('#------------------#')
                if y > lock_size :
                    break
            
print('계산끝')


