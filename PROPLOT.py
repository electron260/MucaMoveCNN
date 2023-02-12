import numpy as np
import pyqtgraph as pg
from collections import deque
from PyQt5 import QtWidgets, QtCore
from pyqtgraph import PlotWidget, plot
import time
import serial 
import signal
import sys
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
from sklearn.cluster import KMeans
from model import Net 
import torch 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

total = []
model = Net().to(device)
model.load_state_dict(torch.load('model_weighs.pth'))
trad = {0: "Slide Left", 1: "Slide Right", 2: "Slide Up", 3: "Slide Down", 4: "Long Touch"}



Listening = False
last = None 
nbpatience = 0
#  on click on widget get x,y coordinates
def SignalHandler(sig, frame):
    print("SignalHandler")

    app.quit()
    
    sys.exit(0)



    

class MainWindow(QtWidgets.QMainWindow):
   
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.graphWidget = pg.PlotWidget()
        
        #add a widget to let you choose between "slide left", "slide right", "no move", "slide up", "slide down", "long touch"
        self.file = None
        self.patience = -1
        

        self.recordmenu = QtWidgets.QPushButton("Action",self)
        self.menu = QtWidgets.QMenu(self)
        self.menu.addAction("Slide Left")
        self.menu.addAction("Slide Right")
        self.menu.addAction("Slide Up")
        self.menu.addAction("Slide Down")
        self.menu.addAction("Long Touch")
        self.recordmenu.setMenu(self.menu)

        #change the value of menu_label when you click on an action
        self.menu.triggered.connect(self.Action)

        self.menu_label = QtWidgets.QLabel("Action : No mode selected")
        #self.menu_label.setText("Action : ", str(self.action))

     
        

        self.Move = "No move"

      


        self.trackingWidget = pg.PlotWidget()

        self.gain = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.gain.setMinimum(0)
        self.gain.setMaximum(100)
        self.gain.setValue(30)

        self.gain.setTickInterval(1)
     
        self.range = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.range.setMinimum(0)
        self.range.setMaximum(500)
        self.range.setValue(30)
        self.range.setTickInterval(1)
        self.range.valueChanged.connect(self.rangeChanged)


        self.treshold = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.treshold.setMinimum(0)
        self.treshold.setMaximum(500)
        self.treshold.setValue(10)
        self.treshold.setTickInterval(1)
        self.treshold.valueChanged.connect(self.tresholdChanged)

        self.infoMove = QtWidgets.QLabel()
        self.infoMove.setText("Move : "+str(self.Move))

        self.gain_label = QtWidgets.QLabel()
        self.gain_label.setText("Gain: " + str(self.gain.value()))
        
        self.treshold_label = QtWidgets.QLabel()
        self.treshold_label.setText("Treshold: " + str(self.treshold.value()))

        self.range_label = QtWidgets.QLabel()
        self.range_label.setText("Range: " + str(self.treshold.value()))

        self.trackingWidget.setBackground('w')





        
        self.graphWidget.setBackground('w')
        self.graphWidget.scene().sigMouseClicked.connect(self.mouseClicked)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.graphWidget)
        layout.addWidget(self.gain)
        layout.addWidget(self.gain_label)
        layout.addWidget(self.treshold)
        layout.addWidget(self.treshold_label)
        layout.addWidget(self.range)
        layout.addWidget(self.range_label)
        layout.addWidget(self.trackingWidget)
        layout.addWidget(self.recordmenu)
        layout.addWidget(self.menu_label)
        layout.addWidget(self.infoMove)

     

        self.gain.valueChanged.connect(self.gainChanged)
        self.setLayout(layout)
        # Create a central widget for the layout
        central_widget = QtWidgets.QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)
    
        # self.slider.setFixedWidth(200)
        # self.slider.setFixedHeight(50
        # self.data_line =  self.graphWidget.plot(self.delegate.times, self.delegate.vals, pen=pen)

        # self.timer2 = QtCore.QTimer()
        # self.timer2.setInterval(10)
        # self.timer2.timeout.connect(self.readByte)
        # self.timer2.start()
        self.timer = QtCore.QTimer()
        self.timer.setInterval(10)
        self.timer.timeout.connect(self.update_plot_data)
        self.timer.start()
        self.ser = serial.Serial('/dev/ttyUSB0', 250000, timeout=0.1)
        self.sync = False
        self.calibration = []
        #TO DEFINE
        self.cols = 9
        self.rows = 9

        #To determine the number of cluesters
        self.model = KMeans(n_clusters=5)
        self.cells = self.cols*self.rows
        self.buffer = np.zeros(self.cells)
        self.mean = 10000
        self.std = 10000

        self.startTime = time.time()

    # def ActionChanged :

 
    def Action(self,action):
        
            self.menu_label.setText("Action : " + action.text())
            self.menu_label.adjustSize()
            files = {"Slide Left": "slideleft.txt", "Slide Right": "slideright.txt", "Slide Up": "slideup.txt", "Slide Down": "slidedown.txt", "Long Touch": "longtouch.txt"}
            self.file = files[action.text()]
            # if self.recordmenu.text() == "Slide Left":
            #     pass
                

    
    def rangeChanged(self):
        range = self.range.value()
        self.range_label.setText("Range: " + str(range))
     

    def tresholdChanged(self):
        treshold = self.treshold.value()
        self.treshold_label.setText("Treshold: " + str(treshold))
      

    def gainChanged(self):
        gain = self.gain.value()
        self.gain_label.setText("Gain: " + str(gain))
        
    def MoveChanged(self):
        self.infoMove.setText("Move : "+str(self.Move))
      

    def mouseClicked(self, event):
        self.calibration = []
        print("Mouse clicked at: ", event.scenePos())

    def readSync(self):
        while(not self.sync):
            data = self.ser.readline()
            if len(data) < 3:
                print("No sync")
                continue
            buffer = data.split(b":")
            if len(buffer) ==5:
                self.cols = int(buffer[1])
                self.rows = int(buffer[3])
                self.cells = self.cols*self.rows
                self.sync = True
                print(self.cols, self.rows, self.cells)
                for i in range(5):
                    print(buffer[i])
                print("Synced")
                return
    
    def readString(self):
        data = self.ser.readline()
        if len(data) >= self.cells:
            data = data.decode("utf-8")
            data = data.split(",")
            try:
                data = [int(i) for i in data]
            except:
                self.sync = False
                self.readSync()
            # print(data)
            if len(data)==self.cells:
                self.buffer = np.array(data)

    def readByte(self):
        data = self.ser.read(self.cells+2)
        if len(data)==self.cells+2:
            lowbyteMin = data[1]
            highbyteMin = data[0]
            min = int(highbyteMin)<<8 + int(lowbyteMin)

            self.buffer = np.zeros(self.cells)
            for i in range(self.cells):
                self.buffer[i] = data[i+2] + min

        else:
            print("No data", data)
            # self.sync = False
            # self.readSync()
            # self.ser.write(b"s")

    def setup(self):
        print("Setup")
        # while not self.sync:
        #     print("Syncing")
        self.readSync()
        self.ser.write(str(self.gain.value()).encode())

    def update_plot_data(self):
            global total, device, nbpatience, last, Listening

            

            gain_bytes = str(self.gain.value()).encode()
            self.ser.write(gain_bytes)
            #self.ser.write(b"s")
            # self.readByte()
            self.readString()
            #  reashaep the data
            self.buffer = self.buffer.reshape(self.rows, self.cols)
            #  plot the data*
            # print(self.buffer.shape)
            temp = self.buffer
            
            if len(self.calibration) <80 and  (time.time() - self.startTime > 2 ):
                self.calibration.append(self.buffer)
                self.mean = np.mean(np.array(self.calibration), axis=0)
                self.std = np.std(np.array(self.calibration), axis=0)
                if len(self.calibration)%10==0:
                    print(len(self.calibration))
    
            # temp = np.clip(temp, 0, 1000)
            temp = temp - self.mean
            #print(temp)
            temp[temp<self.treshold.value()] = 0


            tracking, moyenne = GravityCenter(self, temp)


            if moyenne>0.2 and Listening == False: 
                Listening = True
                
                

            if  Listening == True and len(total)<20:
                total.append(temp)


            if Listening == True and len(total)==20 :    
                inf = inferenceCNN(model, torch.Tensor(total))
                print("inference : ", inf)
                self.Move = trad[inf]
                self.MoveChanged()
                Listening = False 
                total = []




            

            if moyenne > 0.2 and self.file != None: 
                   #change the color of the menu label when the action is detected
                self.patience = 1
            else:
                self.patience -= 1
            if self.patience > 0:
                self.write(temp)
                self.menu_label.setStyleSheet("QLabel { background-color : green; color : white; }")
            elif self.patience == 0:
                self.menu_label.setStyleSheet("QLabel { background-color : white; color : black; }")
                with open(str(self.file), "a") as f:
                    f.write(";\n")

      
            
            #disable auto levels

            image = pg.ImageItem(temp, autoLevels=False, autoRange=False, autoHistogramRange=False, levels=(0, self.range.value()))
            ImageTrack = pg.ImageItem(tracking, autoLevels=False, autoRange=False, autoHistogramRange=False, levels=(0, 1)) 
            self.trackingWidget.plotItem.clear()
            self.trackingWidget.plotItem.addItem(ImageTrack)
            self.graphWidget.plotItem.clear()
            self.graphWidget.plotItem.addItem(image)
            # self.graphWidget.setImage(self.buffer, autoLevels=False, autoRange=False, autoHistogramRange=False, levels=(0, 1000))
        
    def write(self, temp):
        with open(str(self.file), "a") as f:
            f.write(str(temp.reshape(1,-1)[0])+"\n")
            print("str : " + str(temp))

def GravityCenter(self, temp : np.ndarray):
    posx,posy = 0,0
    center = np.zeros((self.rows, self.cols))
    moyenne = np.mean(temp)  
    #print("moyenne value : ", moyenne)
    if moyenne > 1 :
        for x in range(self.cols):
            for y in range(self.rows):
                #print("somme : ",temp.sum(), " type : ", type(temp.sum()))
                posx += (1/temp.sum())*temp[x,y]*x
            
                posy += (1/temp.sum())*temp[x,y]*y
        
        center[int(posx),int(posy)] = 1
    return center, moyenne

def inferenceCNN(model, frames):
    #take a list of frames and return the prediction of the model
    #return the prediction
    #load the model
    frames = frames.to(device)
    frames = frames.unsqueeze(0)
    output = model(frames)
    pred = torch.softmax(output, dim=1).argmax(dim=1)

    return int(pred)

def ElbowCluesters(self,temp : np.ndarray):
    #determine the number of cluesters in temp matrix with the elbow method 
    #return the number of cluesters
    elbowtracking = np.zeros((self.rows, self.cols))
    #get the index of each element not null of temp
    temp = temp.reshape(1,-1)
    temp = temp[0]
    points = np.array([[x%self.cols,x//self.rows] for x in range(len(temp)) if temp[x]!=0])
    print(points)

    visualizer = KElbowVisualizer(self.model, k=5)
    visualizer.fit(points)
    #visualizer.show()
    nbclusters = visualizer.elbow_value_
    print("Nb of cluesters : ",nbclusters)
    if nbclusters != None : 
        kmeans_model = KMeans(n_clusters=int(nbclusters)).fit(points)
        kmeans_model.fit(points)
        #find position of the center of the cluster 
        for i in kmeans_model.cluster_centers_:
            elbowtracking[int(i[0]),int(i[1])] = 1

    
    # kmeans_model.fit(points)


    return elbowtracking, nbclusters
    



#attach the signal
signal.signal(signal.SIGINT, SignalHandler)



app = QtWidgets.QApplication([])
w = MainWindow()
w.setup()
w.show()
app.exec_()
# dev.disconnect()s
while True:
    # if dev.waitForNotifications(1.0):
        # handleNotification() was called
        # continue
    # print("Waiting...")
    pass