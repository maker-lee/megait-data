import googletrans

translator = googletrans.Translator()

read_file_path =r"읽을 영어 파일 절대경로.txt"
write_file_path = r"파일명.txt"

with open(read_file_path, 'r') as f :
    readLines = f.readlines()

for lines in readLines:
    result1 = translator.translate(lines, dest='ko')
    print(result1.text)
    with open(write_file_path,'a', encoding='UTF8') as f:       # 'a' 옵션은 마지막에 추가로 쓰는 모드
        f.write(result1.text + '\n')