import cv2
import numpy as np
import math
from scipy.spatial import distance
import vector
from keras import models
import neur


def main(filename):

   video = cv2.VideoCapture(filename)
   frmCount = 0
   ukupnaSuma = 0
   #ova lista predstavlja sve brojeve koji su bili na sceni
   all = []
   #stari nacin
   #model = models.load_model('model.h5')
   classifier = neur.napravi_model((28, 28, 1), 10)
   classifier.load_weights(''
                             'model.h5')


   linija = nadjiLiniju(filename)
   print(linija)

   minTacka = linija[0]
   maxTacka = linija[1]

   while True:
        ret, frame = video.read()

        if not ret:
            break

        lista_kontura = konture(frame)
        #ovo je bilo za stari nacin
        #kropovani = crop(frame, lista_kontura)

        for kontura in lista_kontura:
            (x, y, w, h) = kontura

            xCenterDot = int(x + w / 2)
            yCenterDot = int(y + h / 2)

            xLeftDot = x
            yLeftDot = y
            xRightDot = x + w
            yRightDot = y + h

            #ovo ce biti recnik koji sadrzi koordinate trenutne konture
            dictNumber = {'dot': (xCenterDot,yCenterDot), 'frameNum': frmCount, 'previousStates': [], 'kontura':kontura}

            closeNumbers = nadjiBroj(all, dictNumber)

            if len(closeNumbers) == 0:

                # dodajemo nove key-value u recnik
                # jos nije prosao liniju
                dictNumber['presaoLiniju'] = False
                # moramo ga dodati u listu poznatih brojeva
                all.append(dictNumber)
                #staro sto nije precizno
                #dictNumber['value'] = predicted(model, kropovani)
                kropovani = prepoznaj(kontura, frame, classifier)
                dictNumber['value'] = kropovani



            elif len(closeNumbers) == 1:

                prev = {'frameNum': frmCount, 'dot': dictNumber['dot'], 'kontura': dictNumber['kontura']}

                #posto je close numbers lista koja ima samo jedan element ovde uzmemo bas indeks broja za koji zelimo da azuriramo
                all[closeNumbers[0]]['previousStates'].append(prev)
                all[closeNumbers[0]]['frameNum'] = frmCount
                all[closeNumbers[0]]['dot'] = dictNumber['dot']
                all[closeNumbers[0]]['kontura'] = dictNumber['kontura']

            #cv2.circle(frame, (xCenterDot, yCenterDot), 5, (0, 0, 255), -1)

            cv2.putText(frame, "Trenutni video: " + str(videoName), (35, 40), cv2.FONT_HERSHEY_COMPLEX,
                        0.5, (0, 255, 255), 1)
            cv2.putText(frame, "Suma: " + str(ukupnaSuma), (35, 80), cv2.FONT_HERSHEY_COMPLEX,
                        0.5, (0, 255, 255), 1)

        for number in all:

            (x,y,w,h) = number['kontura']
            width = int(video.get(3))
            height = int(video.get(4))

            if x < width and y < height:
                if number['presaoLiniju']:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                else:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)


            #for prev in number['previousStates']:
                #if frmCount - prev['frameNum'] < 40:
                    #cv2.circle(frame, prev['dot'], 1, (0, 255, 255), 1)

            #ako nije presao liniju racunamo udaljenost ako je blizu setujemo da je presao
            if not number['presaoLiniju']:
                distanca, _, r = vector.pnt2line(number['dot'], minTacka, maxTacka)
                if distanca < 10.0 and r == 1:
                    if not number['value'] == None:
                        ukupnaSuma += int(number['value'])
                        print(ukupnaSuma)
                    number['presaoLiniju'] = True

        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frmCount += 1

   video.release()
   cv2.destroyAllWindows()

   dopisiSumuUFajl(filename, ukupnaSuma)
"""
pomocna metoda koja upisuje konacnu sumu u fajl
"""
def dopisiSumuUFajl(filename, ukupnaSuma):
    f = open("out.txt", "a+")
    f.write(filename + "\t" +str(ukupnaSuma) + "\n")
    f.close()
"""
metoda koja poziva prepoznavanje broja i prethodno njegovo isecanje na dimenziju 28x28
"""
def prepoznaj(kontura, frame, classifier):
    (x, y, w, h) = kontura
    xCenterDot = int(x + w / 2)
    yCenterDot = int(y + h / 2)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    extra = 12
    number = gray[yCenterDot - extra:yCenterDot + extra, xCenterDot - extra:xCenterDot + extra]
    #cv2.imshow("broj", number)
    _, number = cv2.threshold(number, 165, 255, cv2.THRESH_BINARY)

    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(number, cv2.MORPH_OPEN, kernel)
    dilation = cv2.dilate(opening, kernel, iterations=1)
    erosion = cv2.erode(opening, kernel, iterations=1)

    #error bez ovoga da li je prazan niz
    if not np.shape(number) == ():

        number = crop_num(number)
        cv2.imshow("number", number)
        num = classifier.predict_classes(number.reshape(1, 28, 28, 1))
       #print(num)
        return int(num)


def crop(frame, lista_kontura):
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    _, threshold = cv2.threshold(grayscale, 165, 255, cv2.THRESH_BINARY)
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
    dilation = cv2.dilate(opening, kernel, iterations=1)
    erosion = cv2.erode(opening, kernel, iterations=1)

    _, contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)



    for kontura in contours:

        [x, y, w, h] = cv2.boundingRect(kontura)

        #(x, y, w, h) = kontura

        xLeftDot = x
        yLeftDot = y
        xRightDot = x + w
        yRightDot = y + h

        cropped = threshold[yLeftDot:yRightDot+1, xLeftDot:xRightDot+1]

        resized = cv2.resize(cropped, (28, 28), interpolation=cv2.INTER_NEAREST)

    #cv2.imshow("cropped", cropped)
    #cv2.imshow("resized", resized)
    #cv2.imshow("straightened", straightened)

    scaled = resized / 255
    flattened = scaled.flatten()

    return np.reshape(flattened, (1, 784))
"""
pomocna metoda koja iseca sliku broja na zadatu dimenziju
"""
def crop_num(number):
        _, grayscale = cv2.threshold(number, 165, 255, cv2.THRESH_BINARY)
        #cv2.imshow("grayscale",grayscale)
        _, konture, _ = cv2.findContours(grayscale, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for kontura in konture:
            [x, y, w, h] = cv2.boundingRect(kontura)
            xLeftDot = x
            yLeftDot = y
            xRightDot = x + w
            yRightDot = y + h
        cropped = number[yLeftDot:yRightDot + 1, xLeftDot:xRightDot + 1]
        cropped = cv2.resize(cropped, (28,28), interpolation=cv2.INTER_AREA)
        return cropped

def predicted(model, img_number):

    predicted_result = model.predict(img_number)
    final_result = np.argmax(predicted_result)

    print(final_result)


    return final_result

"""
metoda koja racuna euklidsko rastojanje izmedju trenutnog i svih ostalih brojeva
u zavisnosti od nje odredjujemo da li je broj taj isti koji pratimo ili je novi broj
"""
def nadjiBroj(all, dict):
    founded = []

    for i, el in enumerate(all):

        (aX, aY) = el['dot']
        (bX, bY) = dict['dot']

        a = np.array((aX, aY))
        b = np.array((bX, bY))

        distanceBoundary  = 20
        distanca = distance.euclidean(a, b)

        if distanca < distanceBoundary:
            founded.append(i)

    return founded

"""
metoda koja nalazi konture brojeva sa prosledjenih frejmova
jedna kontura opisana preko 4 komponente x,y,w,h
"""
def konture(frame):

        lista_kontura = []
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        _, threshold = cv2.threshold(grayscale, 165, 255, cv2.THRESH_BINARY)
        kernel = np.ones((1, 1), np.uint8)
        opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)

        #retr_external daje samo spoljne a ne one unutrasnje npr 8 dva kruga
        _, contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for kontura in contours:

            (x, y, w, h) = cv2.boundingRect(kontura)
            coord = (x,y,w,h)
            lista_kontura.append(coord)
            #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            xLeftDot = x
            yLeftDot = y
            xRightDot = x + w
            yRightDot = y + h
            #print(xRightDot)
            #print(yRightDot)

            #cv2.circle(frame,(xLeftDot,yLeftDot),5,(0,0,255),-1)

            #cv2.circle(frame,(xRightDot,yRightDot),5,(0,0,255),-1)

        #cv2.drawContours(frame, contours, -1, (0, 0, 255), 2)

        #cv2.imshow("grayscale", grayscale)
        cv2.imshow("threshold", threshold)

        return lista_kontura

"""
metoda koja nalazi liniju na prvom frejmu video zapisa
koristi canny edge algoritam i HoughLinesTransform na
kraju uzmemo min i max tacku
"""
def nadjiLiniju(filename):
    #treba prepraviti da bude za svaki video
    cap = cv2.VideoCapture(filename)
    #setuje CV_CAP_PROP_POS_FRAMES na taj frame
    cap.set(1, 0)
    _, frame = cap.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    light_blue = np.array([110, 50, 50])
    dark_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, light_blue, dark_blue)
    #cv2.imshow('mask', mask)
    #prima sliku donji i gornji threshold
    edges = cv2.Canny(mask, 75, 150)
    #cv2.imshow('edges',edges)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180,50, maxLineGap=50)

    nizX1 = []
    nizX2 = []
    nizY1 = []
    nizY2 = []
    if lines is not None:
        for line in lines:
        # jer je linija zabelezena kao niz niza a taj podniz ima 4 elementa
            x1, y1, x2, y2 = line[0]
            nizX1.append(x1)
            nizY1.append(y1)
            nizX2.append(x2)
            nizY2.append(y2)


    x1 = min(nizX1)
    y1 = nizY1[nizX1.index(min(nizX1))]
    x2 = max(nizX2)
    y2 = nizY2[nizX2.index(max(nizX2))]
    nadjenaLinija = ((x1, y1), (x2, y2))

    #cv2.line(frame, (x1, y1),(x2, y2), (0, 0, 255), 3)
    #cv2.circle(frame, (min(nizX1), nizY1[nizX1.index(min(nizX1))]), 8, (0, 255, 0), -1)
    #cv2.circle(frame, (max(nizX2), nizY2[nizX2.index(max(nizX2))]), 8, (0, 255, 0), -1)
    #cv2.imshow("frame", frame)

    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return nadjenaLinija

if __name__ == "__main__":

    f = open("out.txt", "w+")
    f.write("RA 199-2014 Vuk Novakovic\n")
    f.write("file	sum\n")
    f.close()
    for i in range(0, 10):
        videoName = 'video-' + str(i) + '.avi'
        main(videoName)

