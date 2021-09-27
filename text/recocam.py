# подключаем библиотеки
import pytesseract # OCR
import cv2 # обработка изображений, computer vision

source = 0
# подключаем модуль
pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"
cap = cv2.VideoCapture(source) # получаем данные с камеры
while(True):
    ret, frame = cap.read()
    string = pytesseract.image_to_string(frame)
    if string:
        print(string)
        target_word = "0"
        data = pytesseract.image_to_data(frame, output_type=pytesseract.Output.DICT)
        tw = target_word
        # for tw in target_word:
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
        string = 0
    cv2.imshow('reco',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture 
cap.release()
cv2.destroyAllWindows()
   


# plt.imsave("all_dog_words.png", frame)
# plt.imshow(frame)
# plt.show()
