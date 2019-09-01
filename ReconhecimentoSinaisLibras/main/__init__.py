#! /usr/bin/python
import cv2
import math
import numpy as np
from time import sleep

class kernel:

    # GLOBALS
    kernel = np.ones((5, 5), np.uint8)
    fgbg = cv2.createBackgroundSubtractorKNN(history=99999999)
    contourAxisAngle = 0
    cogPt = None
    gestures = []
    
    # SETTINGS
    LEARNING_RATE = 0.0005
    TIME_GESTURE = 0.5
    SMALLEST_AREA = 25000
    MAX_POINTS = 20
    MAX_FINGER_TIPS = 5
    MIN_FINGER_DEPTH = 20
    MIN_FINGER_ANGLE = 1
    MAX_FINGER_ANGLE = 60
    MIN_FINGER_DISTANCE = 140
    MAX_FINGER_DISTANCE = 300
    MIN_DISTANCE_BETWEEN_TIPS = 40
    MAX_PRINT_GESTURES = 13
    MAX_Y_ALLOWED = 400
    
    
       
    # @Function findBiggestContour
    # @Description Encontra o melhor contorno na imagem
    # @Author Danilo Dorotheu
    # @Author Marcio F. Cruz
    # @Author Diego Santana
    # @Author Thiago Guy
    # @Param image Imagem tratada pelos filtros
    # @Param output Imagem de saida para ser desenhada
    # @Returns bigContour Maior contorno encontrado
    def findBiggestContour(self, image, output):
        
        # Armazena o maior contorno
        bigContour = None
        
        # Cria uma imagem auxiliar para o algoritmo de encontrar contornos
        imageContour = image.copy()
        
        # Gera todos os contornos
        _, contours, hierarchy = cv2.findContours(imageContour, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Referencia o maior contorno contido na lista
        maxArea = self.SMALLEST_AREA
        
        # Busca o maior contorno dentre aqueles que foram obtidos
        for i in range(len(contours)):
            
            # Referencia os contornos atuais
            cnt = contours[i]
            
            # Algoritmo que calcula a area dos contornos
            area = cv2.contourArea(cnt)
            
            # Verifica se a area obtida e maior que o limite minimo definido
            if(area > self.SMALLEST_AREA):
                
                # Verifica se a area obtida e maior que a area definida como maior
                if(area > maxArea):
                    
                    # Uma nova area maior e encontrada e definida como a maior
                    maxArea = area
                    
                    # Os contornos serao armazenados como sendo os maiores, desde que sejam os maiores da lista
                    bigContour = contours[i]
                    
        # Desenha o contorno da mao
        # cv2.drawContours(output, [bigContour], 0, (255, 0, 0), 2)
                    
        # Retorna o maior contorno da lista
        return bigContour
    
    
    
    # @Function extractContourInfo
    # @Description Extrai COG e calcula angulacao
    # @Author Danilo Dorotheu
    # @Author Marcio F. Cruz
    # @Author Diego Santana
    # @Author Thiago Guy
    # @Param bigContour Maior contorno encontrado que referencia o objeto de interesse
    # @Param output Imagem de saida para ser desenhada
    def extractContourInfo(self, bigContour, output):
        
        # Algoritmo que constroi linhas de convexidade em volta do objeto de interesse
        hull = cv2.convexHull(bigContour)
        
        # Calcula todos os momentos ate a terceira ordem de um poligono ou forma rasterizada
        moments = cv2.moments(bigContour)
        
        # Define o centro da area
        if moments['m00'] != 0:
            cx = int(moments['m10'] / moments['m00'])  # cx = M10/M00
            cy = int(moments['m01'] / moments['m00'])  # cy = M01/M00
        self.cogPt = (cx, cy)
        
        # Desenha um circulo no centro da area
        cv2.circle(output, self.cogPt, 5, [0, 255, 255], 3)
        
        # Desenha o maior contorno encontrado
        # cv2.drawContours(output, [bigContour], 0, (0, 255, 0), 2) 
        
        # Desenha a convexidade do maior contorno encontrado
        cv2.drawContours(output, [hull], 0, (255, 0, 0), 2) 
        
        # Pega os elementos necessarios para obter o angulo da imagem
        m11 = moments['m11']
        m20 = moments['m20']
        m02 = moments['m02']
        
        # Pega o angulo da imagem
        self.contourAxisAngle = self.calculateTilt(m11, m20, m02)

 

    # @Function findFingerTips
    # @Description Encontra os pontos das pontas de dedo
    # @Author Danilo Dorotheu
    # @Author Marcio F. Cruz
    # @Author Diego Santana
    # @Author Thiago Guy
    # @Param bigContour Maior contorno que referencia o objeto de interesse
    # @Param output Imagem de saida para ser desenhada
    # @Returns fingerTips Lista de pontas de dedo encontrada
    def findFingerTips(self, bigContour, output):
        
        # Armazenam os pontos dos defeitos de convexidade
        startTip = []
        endTip = []
        farTip = []
        
        # Reduz o numero de pontos no contorno
        cnt = cv2.approxPolyDP(bigContour, 0.01 * cv2.arcLength(bigContour, True), True)
        
        # Desenha contornos
        # cv2.drawContours(self.drawing, [cnt], 0, (0, 255, 0), 3)
        # Algoritmo que constroi linhas de convexidade em volta do objeto de interesse
        # hull = cv2.convexHull(cnt)
        # cv2.drawContours(self.drawing, [hull], 0, (0, 0, 255), 2) 
        
        # Algoritmo que dara sequencia ao algoritmo de defeito de convexidade
        hull = cv2.convexHull(cnt, returnPoints=False)
        
        # Algoritmo que busca defeitos de convexidade a partir dos contornos do objeto e de sua convexidade
        defects = cv2.convexityDefects(cnt, hull)
        
        # Verifica se existem defeitos de convexidade
        if (defects is None):
            return
        
        # Pega o total de defeitos de convexidade
        defectsTotal = defects.shape[0]
        
        # Verifica se ultrapassa o limite de pontos definido
        if (defectsTotal > self.MAX_POINTS):
            
            # Define o total de defeitos para o valor maximo definido
            defectsTotal = self.MAX_POINTS

        # Percorre o total de defeitos de convexidade
        for i in range(defectsTotal):
            
            # Referenciam os pontos de defeito de convexidade
            s, e, f, d = defects[i, 0]
            
            # Pega o ponto do incio
            start = tuple(cnt[s][0]);
            startTip.append(start)
            
            # Pega o ponto do fim
            end = tuple(cnt[e][0])
            endTip.append(end)
            
            # Pega o ponto distante
            far = tuple(cnt[f][0])
            farTip.append(far)
            
            # Desenha um circulo nos pontos dos defeito de convexidade
            # cv2.circle(output, start, 5, [0, 255, 255], 2)
            # cv2.circle(output, end, 5, [255, 255, 0], 2)
            # cv2.circle(output, far, 5, [255, 0, 255], 2)
            
            # Desenha linhas partindo centro da mao ate as pontas dos dedos
            # cv2.line(output, self.cogPt, start, [0, 255, 255], 2)
            
        # Retorna as possiveis pontas de dedo
        return self.reduceTips(defectsTotal, startTip, endTip, farTip, output)
    
    
    
    # @Function reduceTips
    # @Description Identifica as pontas dos dedos e reduz a quantidade de pontos
    # @Author Danilo Dorotheu
    # @Author Marcio F. Cruz
    # @Author Diego Santana
    # @Author Thiago Guy
    # @Param numPoints Numero de pontos dos defeito de convexidade
    # @Param start Lista de pontos do inicio
    # @Param end Lista de pontos do fim
    # @Param far Lista de pontos distantes
    # @Param output Imagem de saida para ser desenhada
    def reduceTips(self, numPoints, start, end, far, output):
        
        # Armazena os dedos reconhecidos
        fingerTips = []
        
        # Percorre a lista de pontos dos defeitos de convexidade
        for i in range(numPoints):
            
            # Verifica se a profundidade ultrapassa o limite definido
            if (far[i] < self.MIN_FINGER_DEPTH):
                continue
            
            #  Ve os pontos de dobragem de ambos os lados da ponta do dedo
            pdx = (numPoints - 1) if (i == 0) else (i - 1)
            sdx = 0 if (i == numPoints - 1) else (i + 1)
            
            # Pega o angulo entre a ponta do dedo e as duas dobras
            angle = self.angleBetween3P(start[i], far[pdx], far[sdx])
            
            # Pega a distancia entre o centro da area e a ponta do dedo
            distance = self.distanceBetween2P(self.cogPt, start[i])
            
            # Verifica se o angulo ultrapassa o limite definido
            if (angle >= self.MAX_FINGER_ANGLE):
                continue
            
            # Verifica se a distancia ultrapassa o limite definido
            if (distance < self.MIN_FINGER_DISTANCE or distance > self.MAX_FINGER_DISTANCE):
                continue
            
            # Verifica se o ponto esta muito baixo na tela
            if (start[i][1] >= self.MAX_Y_ALLOWED):
                continue
            
            # Referencia o total de ponta de dedos encontradas ate o momento
            numFigerTips = len(fingerTips)
            
            # Verifica se ja foi encontrada alguma ponta de dedo
            if (numFigerTips > 0):
                
                # Pega a distancia entre a ponta do dedo atual e de seu antecessor
                distanceBetweenPrevious = self.distanceBetween2P(start[i], fingerTips[numFigerTips - 1])
                
                # Verifica se a ponta do dedo anterior esta muito proxima
                if (distanceBetweenPrevious < self.MIN_DISTANCE_BETWEEN_TIPS):
                    continue
            print distance
            # Este ponto provavelmente e um dedo, portanto sera adicionado a lista
            fingerTips.append(start[i])
            
            # Adiciona mais uma quantidade ao total de ponta de dedos encontrada
            numFigerTips += 1
            
            # Desenha um ponto que representa o dedo
            cv2.circle(output, start[i], 5, [255, 255, 255], -5)
            
            # Desenha uma linha entre o centro da area e a ponta do dedo
            cv2.line(output, self.cogPt, start[i], [0, 255, 0], 2)
            
            # Desenha o numero que representa o dedo
            cv2.putText(output, str(numFigerTips), (start[i][0], start[i][1] - 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255))
            
            # Se as 5 pontas forem encontradas, a lista de ponta de dedos e retornada
            if (numFigerTips == self.MAX_FINGER_TIPS):
                return fingerTips
            
        # Retorna a lista de ponta de dedos encontrada
        return fingerTips
    
    
    
    # @Function recognizeHandGesturesInAlphabetPounds
    # @Description Reconhece um gesto do alfabeto de libras
    # @Author Danilo Dorotheu
    # @Author Marcio F. Cruz
    # @Author Diego Santana
    # @Author Thiago Guy
    # @Param fingerTips Lista de pontos de possiveis pontas de dedo
    # @Param output Imagem de saida para ser desenhada
    def recognizeHandGesturesInAlphabetPounds(self, fingerTips, output):

        # Verifica se a busca pelas pontas de dedo foi iniciada
        if (fingerTips is None):
            return
        
        # Referencia o total de ponta de dedos encontrada
        numFigerTips = len(fingerTips)
        
        # Referencia gesto reconhecido
        gesture = None
        
        # Verifica se foi encontrado 2 pontas de dedo
        if (numFigerTips == 2):
            
            # Pega o angulo entre o centro da area e as 2 pontas de dedo
            angle1 = self.angleToCOG(fingerTips[0], self.cogPt, self.contourAxisAngle)
            angle2 = self.angleToCOG(fingerTips[1], self.cogPt, self.contourAxisAngle)
            angle = self.differenceBetween2A(angle1, angle2)
            
            # Verifica se o angulo esta entre 30 e 60
            if (angle >= 30 and angle <= 60):
                
                # Verifica se o primeiro ponto esta abaixo e outro acima do centro da area
                if (fingerTips[0][1] < self.cogPt[1] - 30 and fingerTips[1][1] > self.cogPt[1] + 20):
                    
                    # Verifica se os 2 pontos estao a direita do centro da area
                    if (fingerTips[0][0] > self.cogPt[0] + 50 and fingerTips[1][0] > self.cogPt[0] + 50):
                        
                        # Gesto C reconhecido
                        gesture = 'C'
                    
                    # Verifica se os 2 pontos estao a esquerda do centro da area
                    elif (fingerTips[0][0] < self.cogPt[0] - 50 and fingerTips[1][0] < self.cogPt[0] - 50):
                        
                        # Gesto C reconhecido
                        gesture = 'C'
                
                # Verifica se o primeiro e segundo ponto estao acima do centro da area
                elif (fingerTips[0][1] < self.cogPt[1] - 100 and fingerTips[1][1] < self.cogPt[1] - 100):
                    
                    # Gesto V reconhecido
                    gesture = 'V'
                    
            # Verifica se o angulo esta entre 80 e 100
            if (angle >= 80 and angle <= 100):
                
                # Verifica se o primeiro ponto esta acima e o outro a esquerda do centro da area 
                if (fingerTips[0][1] < self.cogPt[1] - 100 and fingerTips[1][0] < self.cogPt[0] - 100):
                    
                    # Gesto L reconhecido
                    gesture = 'L'
                    
                # Verifica se o segundo ponto esta acima e o outro a direita do centro da area 
                elif (fingerTips[1][1] < self.cogPt[1] - 100 and fingerTips[0][0] > self.cogPt[0] + 100):
                    
                    # Gesto L reconhecido
                    gesture = 'L'
        
        # Desenha o titulo GESTO e o gesto caso seja reconhecido
        cv2.putText(output, 'GESTO: ' + ('' if (gesture is None) else gesture), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5, cv2.LINE_AA)
        
        # Retorna o gesto reconhecido
        return gesture
    
    
    
    # @Function printGestures
    # @Description Desenha um historico dos ultimos gestos reconhecidos
    # @Author Danilo Dorotheu
    # @Author Marcio F. Cruz
    # @Author Diego Santana
    # @Author Thiago Guy
    # @Param gesture Gesto reconhecido
    # @Param output Imagem de saida para ser desenhada
    def printGestures(self, gesture, output):
        
        # Verifica se o gesto foi reconhecido
        if (gesture is not None):
            
            # Referencia o total de gestos contidos no historico
            numGestures = len(self.gestures)
            
            # Verifica se o total de gestos no historico e igual ao limite definido
            if (numGestures == self.MAX_PRINT_GESTURES):
                
                # Limpa a lista caso tenha atingido seu limite definido
                self.gestures = []
                
            # Adiciona um gesto a lista de historico de gestos
            self.gestures.append(gesture)
                
        # Referencia o total de gestos contidos no historico mais o novo gesto
        numGestures = len(self.gestures)
        
        # Desenha a quantidade de gestos no historico
        cv2.putText(output, 'Total: ' + str(numGestures), (40, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3, cv2.LINE_AA)
        
        # Percorre a lista de historico de gestos e os desenha
        for i in range(len(self.gestures)):
            cv2.putText(output, self.gestures[i], ((i + 1) * 45, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3, cv2.LINE_AA)
    
    
    # @Function angleBetween3P
    # @Description Calcula o angulo entre os dedos
    # @Author Danilo Dorotheu
    # @Author Marcio F. Cruz
    # @Author Diego Santana
    # @Author Thiago Guy
    # @Param p1 Ponto da ponta do dedo
    # @Param p2 Ponto antecessor da ponta do dedo
    # @Param p3 Ponto posterior da ponta do dedo
    # @Param output Imagem de saida para ser desenhada
    def angleBetween3P(self, p1, p2, p3):
        
        # Elimina a possibilidade de resultados negativos
        x = (p1[0] - p2[0]) if (p1[0] > p2[0]) else (p2[0] - p1[0])
        y = (p1[1] - p2[1]) if (p1[1] > p2[1]) else (p2[1] - p1[1])
        z = (p1[0] - p3[0]) if (p1[0] > p3[0]) else (p3[0] - p1[0])
        w = (p1[1] - p3[1]) if (p1[1] > p3[1]) else (p3[1] - p1[1])
        
        # Calcula a tangente do X e Y
        tanXY = math.atan2(x, y)
        
        # Calcula a tangente do Z e W
        tanZW = math.atan2(z, w)
        
        # Calcula o angulo das tangentes
        angle = tanXY - tanZW if (tanXY > tanZW) else tanZW - tanXY
        
        # Converte o angulo radiano para grau, arredonda e pega o valor absoluto
        return math.fabs(int(round(math.degrees(angle))))
    
    
    
    # @Function distanceBetween2P
    # @Description Calcula a distancia entre os dedos
    # @Author Danilo Dorotheu
    # @Author Marcio F. Cruz
    # @Author Diego Santana
    # @Author Thiago Guy
    # @Param p1 Primeiro ponto
    # @Param p2 Segundo ponto
    def distanceBetween2P(self, p1, p2):
        
        # Elimina a possibilidade de resultados negativos
        x = (p1[0] - p2[0]) if (p1[0] > p2[0]) else (p2[0] - p1[0])
        y = (p1[1] - p2[1]) if (p1[1] > p2[1]) else (p2[1] - p1[1])
        
        # Calcula a distancia entre os 2 pontos
        return math.hypot(x, y)
    
    
    
    # @Function differenceBetween2A
    # @Description Calcula a diferenca entre 2 angulos
    # @Author Danilo Dorotheu
    # @Author Marcio F. Cruz
    # @Author Diego Santana
    # @Author Thiago Guy
    # @Param a1 Primeiro angulo
    # @Param a2 Segundo angulo
    def differenceBetween2A(self, a1, a2):
        
        # Elimina a possibilidade de resultados negativos
        angle = (a1 - a2) if (a1 > a2) else (a2 - a1)
        
        # Verifica se existe algum angulo negativo e recalcula
        angle = 360 - angle if (a1 < 0 or a2 < 0) else angle
        
        # Retorna o angulo da diferenca de 2 angulos
        return angle
    
    
    
    # @Function angleToCOG
    # @Description Calcula o angulo entre os dedos
    # @Author Danilo Dorotheu
    # @Author Marcio F. Cruz
    # @Author Diego Santana
    # @Author Thiago Guy
    # @Param tipPt Ponto da ponta do dedo
    # @Param cogPt Ponto do centro da area do maior contorno
    # @Param contourAxisAngle Angulo do eixo do contorno
    def angleToCOG(self, tipPt, cogPt, contourAxisAngle):
        
        # Calcula a diferenca do Y entre o ponto do centro da area e a ponta do dedo
        yOffset = cogPt[1] - tipPt[1]
        
        # Calcula a diferenca do X entre o ponto do centro da area e a ponta do dedo
        xOffset = tipPt[0] - cogPt[0]
        
        # Calcula o teta de X e Y
        theta = math.atan2(yOffset, xOffset)
        
        # Converte o teta de radiano para grau e o arredonda
        angleTip = int(round(math.degrees(theta)))
        
        # Retorna o angulo do dedo mais o angulo do eixo do contorno subtraido por um angulo de 90
        # Isso assegura que a mao esteja orientada para cima
        return angleTip + (90 - contourAxisAngle)
    

    
    # @Function calculateTilt
    # @Description Calcula o angulo do eixo do contorno
    # @Author Danilo Dorotheu
    # @Author Marcio F. Cruz
    # @Author Diego Santana
    # @Author Thiago Guy
    # @Param m11 Elemento [m11] da lista do moments
    # @Param m20 Elemento [m20] da lista do moments
    # @Param m02 Elemento [m02] da lista do moments
    def calculateTilt(self, m11, m20, m02):
        
        # Pega a diferenca de 2 elementos
        diff = m20 - m02
        
        # Verifica se nao ha diferenca
        if (diff == 0):
            
            # Retorna o angulo 0 caso o primeiro elemento seja igual a 0
            if(m11 == 0):
                return 0
            
            # Retorna o angulo 45 caso o primeiro elemento seja maior que 0
            elif (m11 > 0):
                return 45
            
            else:
                # Retorna o angulo 45 negativado caso seu valor seja igual ou menor que 0
                return -45
        
        # Pega o teta calculando a tangente do primeiro elemento com a diferenca
        theta = 0.5 * math.atan2(2 * m11, diff)
        
        # Converte o angulo radiano para grau e o arredonda
        tilt = int(round(math.degrees(theta)))
        
        # Verifica algumas condicoes para retornar o angulo esperado
        if (diff > 0 and m11 == 0):
            return 0
        elif (diff < 0 and m11 == 0):
            return -90
        elif (diff > 0 and m11 > 0):  # 0 a 45
            return tilt
        elif (diff > 0 and m11 < 0):  # -45 a 0
            return 180 + tilt  # Transforma em um angulo anti-horario
        elif (diff < 0 and m11 > 0):  # 45 a 90
            return tilt  # Transforma em um angulo anti-horario
        elif (diff < 0 and m11 < 0):  # -90 a -45
            return 180 + tilt
        
        # Nao foi possivel calcular o angulo, portanto retorna o angulo 0
        print 'ERROR TO CALCULATE THE TILT'
        return 0
    

    
    # @Function update
    # @Description Atualiza os processos aplicando tecnicas e renderizando a tela
    # @Author Danilo Dorotheu
    # @Author Marcio F. Cruz
    # @Author Diego Santana
    # @Author Thiago Guy
    # @Param image Matriz de pixels de uma imagem
    def update(self, image):
        
        # Cria uma nova imagem pra exibir os objetos encontrados na tela
        drawing = np.zeros(image.shape, np.uint8)
        
        # Reduz ruidos e detalhes da imagem
        blur = cv2.GaussianBlur(image, (5, 5), 0)
        
        # Utiliza um algoritmo baseado em aprendizagem para subtrair o fundo
        # Aplica um filtro de limiarizacao em binario
        bgSub = self.fgbg.apply(blur, learningRate=self.LEARNING_RATE)
        
        # Limpa os ruidos do fundo
        # Erosao seguida da dilatacao
        opening = cv2.morphologyEx(bgSub, cv2.MORPH_OPEN, self.kernel)
        
        # Limpa os ruidos do objeto
        # Dilatacao seguida da erosao
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, self.kernel)

        # Procura o maior contorno (ex. entre 2 maos, a maior ou a que esta a sua frente sera selecionada)
        bigContour = self.findBiggestContour(closing, drawing)
        
        # Verifica se ha um contorno grande
        if (bigContour is None):
            return
        
        # Pega o centro da area do maior contorno encontrado
        self.extractContourInfo(bigContour, drawing)
        
        # Procura pelas ponta dos dedos
        fingerTips = self.findFingerTips(bigContour, drawing)
        
        # Reconhece o gesto no alfabeto de libras
        gesture = self.recognizeHandGesturesInAlphabetPounds(fingerTips, drawing)
        
        # Desenha um historico dos ultimos gestos reconhecidos
        self.printGestures(gesture, drawing)
            
        # Verifica se algum gesto foi reconhecido
        if (gesture is not None):
            
            # Congela a tela por um tempo definido
            sleep(self.TIME_GESTURE)
        
        # Renderiza a imagem original na tela
        cv2.imshow('UNIP - TCC - Reconhecimento de Gestos do Alfabeto de Libras - Imagem Original', image)
        
        # Renderiza a imagem processada na tela
        cv2.imshow('UNIP - TCC - Reconhecimento de Gestos do Alfabeto de Libras - Imagem Processada', drawing)
        

# Verifica o nome do arquivo
if __name__ == "__main__":
    
    # Instancia a classe Kernel
    kernel = kernel()
    
    # Pega a captura da camera
    cap = cv2.VideoCapture(0)
    
    # O processo e repetido enquanto a camera estiver operando
    while(cap.isOpened()):
        
        # Obtem uma matriz de pixels da imagem capturada pela camera
        ret, image = cap.read()
        
        # Aplica todas as tecnicas e renderiza na tela
        kernel.update(image)
        
        # Interrompe se o ESC for pressionado
        k = cv2.waitKey(10)
        if k == 27:
            break
