# подключаем библиотеки
import pytesseract # OCR
import cv2 # обработка изображений, computer vision

frame = cv2.imread("0.png") # открываем и читаем картинку
# подключаем модуль
pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"
string = pytesseract.image_to_string(frame) # запуск OCR
print(string) # вывод текста

target_word = ['a',"o"]
data = pytesseract.image_to_data(frame, output_type=pytesseract.Output.DICT)

for tw in target_word:
    word_occurences = [ i for i, word in enumerate(data["text"]) if word.lower() == tw ]
    for occ in word_occurences:
        w = data["width"][occ]
        h = data["height"][occ]
        l = data["left"][occ]
        t = data["top"][occ]
        p1 = (l, t)
        p2 = (l + w, t)
        p3 = (l + w, t + h)
        p4 = (l, t + h)
        frame = cv2.line(frame, p1, p2, color=(255, 0, 0), thickness=2)
        frame = cv2.line(frame, p2, p3, color=(255, 0, 0), thickness=2)
        frame = cv2.line(frame, p3, p4, color=(255, 0, 0), thickness=2)
        frame = cv2.line(frame, p4, p1, color=(255, 0, 0), thickness=2)
        
plt.imshow(frame)
plt.show()