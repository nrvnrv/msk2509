# подключаем библиотеки
import pytesseract # OCR
import cv2 # обработка изображений, computer vision

image = cv2.imread("data/0.png") # открываем и читаем картинку
# подключаем модуль
pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"
string = pytesseract.image_to_string(image) # запуск OCR
print(string) # вывод текста
