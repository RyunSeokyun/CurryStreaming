# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import socket
import sys
import numpy as np
import struct
import matplotlib.pyplot as plt
import time
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtCore import *
from scipy import signal
import pickle
import time


 
    
class Thread1(QThread):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        
    def run(self):   
        
        
        sbuff = bytes(b'CTRL\x00\x02\x00\x08\x00\x00\x00')
        BMI.sock.send(sbuff)
        print('Send {0}'.format(sbuff))
        
        time.sleep(0.05)
        
        #while 1:
        received = b""
        while len(received) < 200 : #It seems CURRY sends data separately (header, body)
            rbuff = BMI.sock.recv(1000000)
            received += rbuff
        #print('Receive : {0}'.format(received))
        ## Unpack header
        BMI.Code = struct.unpack('>H', received[4:6])
        BMI.Code = BMI.Code[0]
        print(BMI.Code)
        BMI.Request = struct.unpack('>H', received[6:8])
        BMI.Request = BMI.Request[0]
        print(BMI.Request)
        BMI.StartSample = struct.unpack('>I', received[8:12])
        BMI.StartSample = BMI.StartSample[0]
        print(BMI.StartSample)
        BMI.PacketSize = struct.unpack('>I', received[12:16])
        BMI.PacketSize = BMI.PacketSize[0]
        print(BMI.PacketSize)
        #print(received)
        
        
        BMI.Samples = int(BMI.PacketSize/BMI.DataSize)
        Result = np.zeros((BMI.Samples,1), dtype = 'float')
        for i in range (0, BMI.Samples):
            testResult = struct.unpack('f', received[20+4*i:20+4*(i+1)])
            Result[i] = testResult[0]
        #print(Result)
        
        
        #print(np.size(Result))
        BMI.nSamples = int(BMI.Samples/BMI.Chan)
        BMI.FinalData = np.zeros((BMI.Chan,BMI.SRate*BMI.BuffSize), dtype = 'float')
        BMI.Final = np.zeros((BMI.Chan,BMI.nSamples), dtype = 'float')
        cnt = 0
        for i in range (0, BMI.nSamples):
            for j in range(0,BMI.Chan):
                BMI.Final[j,i] = Result[cnt]
                cnt = cnt +1
        
        ### Filter Coeff
        print("Calculating Filter Coefficient...")
        sos = signal.ellip(5, 0.009, 80, 100/(BMI.SRate/2),btype = 'low',analog = False, output = 'sos')
        zi = signal.sosfilt_zi(sos)
        ZI = np.zeros((3,2,BMI.Chan))
        for i in range(0, BMI.Chan):
            ZI[:,:,i] = zi
        zi = ZI
        zf = ZI
        print("Done..\n")
        ## Filtering
        for i in range(0, BMI.Chan):
            BMI.Final[i,:], zf[:,:,i] = signal.sosfilt(sos, BMI.Final[i,:],-1,zi[:,:,i])
        zi = zf
        
        BMI.fileSave(BMI, 1)
        print(BMI.SaveName)
        BMI.FinalData[:,:BMI.SRate*BMI.BuffSize-BMI.nSamples] = BMI.FinalData[:,BMI.nSamples:BMI.SRate*BMI.BuffSize]
        BMI.FinalData[:,BMI.SRate*BMI.BuffSize-BMI.nSamples:BMI.SRate*BMI.BuffSize] = BMI.Final
        
        #plt.plot(Final[0,:])
        #plt.show()
        
        Result = np.zeros((BMI.Samples,1), dtype = 'float')
        BMI.Final = np.zeros((BMI.Chan,BMI.nSamples), dtype = 'float')

        
        # Streaming Loop
        while 1 :
        
            
            received = b""
            while len(received) < 200 : 
                rbuff = BMI.sock.recv(1000000)
                received += rbuff
        
            BMI.Trig = 0
            
            for i in range (0, BMI.Samples):
                testResult = struct.unpack('f', received[20+4*i:20+4*(i+1)])
                Result[i] = testResult[0]
            #print(Result)
        
        
            #print(np.size(Result))
            
            cnt = 0
            for i in range (0, BMI.nSamples):
                for j in range(0,BMI.Chan):
                    BMI.Final[j,i] = Result[cnt]
                    cnt = cnt +1
            
            ## Filtering
            for i in range(0, BMI.Chan):
                BMI.Final[i,:], zf[:,:,i] = signal.sosfilt(sos, BMI.Final[i,:],-1,zi[:,:,i])
            zi = zf
            
            
            BMI.fileSave(BMI, 1)
            BMI.FinalData[:,:BMI.SRate*BMI.BuffSize-BMI.nSamples] = BMI.FinalData[:,BMI.nSamples:BMI.SRate*BMI.BuffSize]
            BMI.FinalData[:,BMI.SRate*BMI.BuffSize-BMI.nSamples:BMI.SRate*BMI.BuffSize] = BMI.Final
            #plt.plot(BMI.Final[0,:])
            #plt.show()
            BMI.Trig = 1
        
            
            if BMI.StopFlg ==1 :
                print("Forced quit\n")
                sbuff = bytes(b'CTRL\x00\x02\x00\x09\x00\x00\x00')
                BMI.sock.send(sbuff)
                print('Send {0}'.format(sbuff))
                break
            
            time.sleep(0.001)


class Thread2(QThread):
    def __init__(self, parent=None):
        super().__init__(parent)
        
    def run(self):

        print("GoGo Thread2\n")

        while 1:
            if BMI.Trig >= 1:
                #FinalData = butter_Filter(FinalData, 3, SRate, 3)
                plt.plot(BMI.FinalData[0,:])
                plt.show()
                #print(BMI.FinalData[0,0:3])
                time.sleep(0.001)
                BMI.Trig = 0
            
            if BMI.StopFlg == 1:
                print("Quit 2\n")
                break








def butter_Filter(data, cutoff, fs, order):
    nyq = 0.5*fs
    normal_cutoff = cutoff/nyq
    b, a = signal.butter(order, normal_cutoff, btype = 'high', analog = False)
    y = signal.filtfilt(b, a, data)
    return y




class BMI(QWidget):
    
    ##### Public Variables ###############
    lock = 0
    BuffSize = 5
    sock = 0
    Version = 803
    Size = 0
    Chan = 10
    SRate = 1000
    DataSize = 4
    Code = 1
    Request = 2
    StartSample = 0
    PacketSize = 4
    Samples = 0
    nSamples = 500
    FinalData = 0
    Final = 0
    Trig = 0
    StopFlg = 0
    fileName = ""
    SaveName = "Data.bin"
    EventName = ""
    PredEvName = ""
    
    t = 0
    t2 = 0
    
    
    def __init__(self):
        super().__init__()
        Streaming = QPushButton("Streaming")
        Streaming.clicked.connect(self.Streaming_pressed)
        
        Training = QPushButton("Training")
        Training.clicked.connect(self.training_button)
        
        DecoderTrain = QPushButton("Decoder Training")
        DecoderTrain.clicked.connect(self.decoder_train_button)
        
        Prediction = QPushButton("Prediction")
        Prediction.clicked.connect(self.prediction_button)
        
        StopSign = QPushButton("Stop")
        StopSign.clicked.connect(self.Stop_pressed)
        
        vbox = QVBoxLayout()
        vbox.addWidget(Streaming)
        vbox.addWidget(Training)
        vbox.addWidget(DecoderTrain)
        vbox.addWidget(Prediction)
        vbox.addWidget(StopSign)
        
        
        self.resize(400,400)
        self.setLayout(vbox)
        
        
    def Streaming_pressed(self):
        print('pressed')
        BMI.StopFlg = 0
        BMI.SaveName = 'Data.bin'
        BMI.ConnectCurry(self)
        BMI.ReceiveBasicInfo(self)
        BMI.fileSave(self, 0)
        
        print("Starting thread 1.\n")
        h1 = Thread1(self)
        h1.start()
        print("Satrting thread 2.\n")
        h2 = Thread2(self)
        h2.start()
        
    def training_button(self):
        print("Training..")
        BMI.StopFlg = 0
        BMI.ConnectCurry(self)
        BMI.ReceiveBasicInfo(self)
        
        tt = BMI.TimeStampFunc(self, 1)
        BMI.SaveName = 'Data' + str(tt) + '.bin'
        BMI.EventName = 'Event' + str(tt) + '.dat'
        print(BMI.SaveName)
        time.sleep(0.1)
        BMI.fileSave(self, 0)
        
        print('Starting thread 1. \n')
        h1 = Thread1(self)
        h1.start()
        print("Starting thread 2. \n")
        h2 = Thread2(self)
        h2.start()
            
        
    def decoder_train_button(self):
        print("Decoder Training..")
        BMI.fileName = BMI.FileOpenFunc(self)
        print(BMI.fileName)
        with open(BMI.fileName, 'rb') as f:
            BMI.Chan = pickle.load(f)
            BMI.SRate = pickle.load(f)
            cnt = 0
            TotData = []
            Stamp = []
            while 1:
                try:
                    data1 = pickle.load(f)
                    data2 = pickle.load(f)
                    Stamp.append(data1)
                    TotData.append(data2)
                    cnt +=1
                except:
                    break
            
        print(BMI.Chan)
        print(BMI.SRate)
        ArrayData = np.zeros((BMI.Chan,int(np.size(TotData)/BMI.Chan)))
        print(np.size(Stamp))
        BuffN = int((np.size(ArrayData)/BMI.Chan)/np.size(Stamp))
        print(BuffN)
        print(TotData[0])
        for i in range(0, np.size(Stamp)):
            ArrayData[:,i*BuffN:(i+1)*BuffN] = TotData[i]
        TotData = ArrayData
        print(TotData)
        plt.plot(TotData[1,1:1000])
        plt.show()
        
        
            
        
    def prediction_button(self):
        print("Prediction")
        BMI.StopFlg = 0
        BMI.ConnectCurry(self)
        BMI.ReceiveBasicInfo(self)
        
        tt = BMI.TimeStampFunc(self, 1)
        BMI.SaveName = 'PredData' + str(tt) + '.bin'
        BMI.PredEvName = 'PredEvent' + str(tt) + 'dat'
        print(BMI.SaveName)
        time.sleep(0.1)
        BMI.fileSave(self, 0)
        
        print('Starting thread 1. \n')
        h1 = Thread1(self)
        h1.start()
        print("Starting thread 2. \n")
        h2 = Thread2(self)
        h2.start()
            
        
        
        
        
    def Stop_pressed(self):
        BMI.StopFlg = 1
        print("Stop")
        
        
        

############ Functions ##############
    def ConnectCurry(self):
        print("Connecting Curry...")
        BMI.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        BMI.sock.connect(('localhost', 4455))
        print("Done..\n")

    def ReceiveBasicInfo(self):
        print("Receiving Basic Info..")
        ## Request Version
        sbuff = bytes(b'CTRL\x00\x02\x00\x01\x00\x00\x00')
        BMI.sock.send(sbuff)
        time.sleep(0.05)
        received = b""
        while len(received) < 19:
            rbuff = BMI.sock.recv(1024)
            received += rbuff
        print(received)
        BMI.Version = struct.unpack('I', received[20:24])
        BMI.Version = BMI.Version[0]
        print(BMI.Version)
        
        ## Request basic info
        sbuff = bytes(b'CTRL\x00\x02\x00\x06\x00\x00\x00')
        BMI.sock.send(sbuff)
        time.sleep(0.05)
        received = b""
        while len(received) < 24:
            rbuff = BMI.sock.recv(1024*8)
            received += rbuff
        print(received)
        
        BMI.Size = struct.unpack('I', received[20:24])
        BMI.Size = BMI.Size[0]
        print(BMI.Size)
        
        BMI.Chan = struct.unpack('I', received[24:28])
        BMI.Chan = BMI.Chan[0]
        print(BMI.Chan)
        
        BMI.SRate = struct.unpack('I', received[28:32])
        BMI.SRate = BMI.SRate[0]
        print(BMI.SRate)
        
        BMI.DataSize = struct.unpack('I', received[32:36])
        BMI.DataSize = BMI.DataSize[0]
        print(BMI.DataSize)
        
        print("Done..\n")
        
    def FileOpenFunc(self):
        fname = QFileDialog.getOpenFileName(self, 'Open File for Training', './')
        return fname[0]
        
       
    def fileSave(self, Mode):
        if Mode == 0 :
            ### Channel, SRate, Timestamp, Data
            with open(self.SaveName, 'wb') as f:
                pickle.dump(self.Chan, f)
                pickle.dump(self.SRate, f)
        else:
            tt = self.TimeStampFunc(self, 0)
            with open(self.SaveName, 'ab') as f:
                pickle.dump(tt, f)
                pickle.dump(self.Final, f)
            
    def TimeStampFunc(self, Mode):
        if Mode ==0:
            t = time.perf_counter_ns()
            tt = int(t/1000)
        else:
            tt = int(time.perf_counter())
        return tt

        
        
        
        



if __name__ == '__main__':
    
    app = QApplication(sys.argv)
    widget = BMI()
    widget.show()
    sys.exit(app.exec_())
    
    print("Exit Main Thread")
    time.sleep(2)
    print("End")
    
