import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5 import QtCore, QtWidgets,QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel,QMainWindow, QTableWidget, QTableWidgetItem, QWidget
from PyQt5.QtWidgets import QFormLayout, QDockWidget, QComboBox, QHBoxLayout, QPushButton, QTextEdit, QAction, QApplication, QDesktopWidget
from PyQt5.QtGui import QIcon
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import sys
import numpy as np
import random
import time
import math

class My_Main_window(QtWidgets.QDialog):
    def __init__(self,parent=None):
        
        super(My_Main_window,self).__init__(parent)

        # set the User Interface
        self.setWindowTitle('NN')
        self.resize(1000, 560)

        # set the figure (left&right)
        self.figure_1 = Figure(figsize=(4, 4), dpi=100)
        self.figure_2 = Figure(figsize=(4, 4), dpi=100)
        self.canvas_1 = FigureCanvas(self.figure_1)
        self.canvas_2 = FigureCanvas(self.figure_2)

        # draw the initial axes of left graph
        self.ax_1 = self.figure_1.add_axes([0.1,0.1,0.8,0.8])
        self.ax_1.set_title("Train Module")
        self.ax_1.set_xlim([-5,5])
        self.ax_1.set_ylim([-5,5])
        self.ax_1.plot()

        # draw the initial axes of right graph
        self.ax_2 = self.figure_2.add_axes([0.1,0.1,0.8,0.8])
        self.ax_2.set_title("Test Module")
        self.ax_2.set_xlim([-5,5])
        self.ax_2.set_ylim([-5,5])
        self.ax_2.plot()

        # set the button
        self.button_test = QPushButton("Test")
        self.button_train = QPushButton("Train")

        # set the combo box
        self.combo = QComboBox()
        self.combo.addItems(["perceptron1.txt", "perceptron2.txt", "2Ccircle1.txt",
                             "2Circle1.txt", "2Circle2.txt", "2CloseS.txt",
                             "2CloseS2.txt", "2CloseS3.txt", "2cring.txt",
                             "2CS.txt", "2Hcircle1.txt", "2ring.txt"])

        # set the left_table that show the valve & key value
        self.ltable = QTableWidget()
        self.ltable.setRowCount(3)
        self.ltable.setColumnCount(4)
        self.ltable.setColumnWidth(0,68)
        self.ltable.setColumnWidth(1,125)
        self.ltable.setColumnWidth(2,125)
        self.ltable.setColumnWidth(3,125)
        self.ltable.verticalHeader().setVisible(0)
        self.ltable.horizontalHeader().setVisible(0)
        self.ltable.setItem(0,0,QTableWidgetItem("Key1"))
        self.ltable.setItem(1,0,QTableWidgetItem("Key2"))
        self.ltable.setItem(2,0,QTableWidgetItem("Key3"))

        # set the right_table that show the train & test rate
        self.rtable = QTableWidget()
        self.rtable.setRowCount(3)
        self.rtable.setColumnCount(2)
        self.rtable.setColumnWidth(0,80)
        self.rtable.setColumnWidth(1,215)
        self.rtable.verticalHeader().setVisible(0)
        self.rtable.horizontalHeader().setVisible(0)
        self.rtable.setItem(0,0,QTableWidgetItem("Train Rate"))
        self.rtable.setItem(1,0,QTableWidgetItem("Test Rate"))
        self.rtable.setItem(2,0,QTableWidgetItem("RMSE"))

        # set the label
        self.label_learn=QLabel()
        self.label_time=QLabel()
        self.label_file=QLabel()
        self.label_learn.setText("Learn Rate")
        self.label_time.setText("Training Time")
        self.label_file.setText("File Name")

        # set the text editor
        self.editor_learn = QtWidgets.QLineEdit()
        self.editor_time = QtWidgets.QLineEdit()

        # the default of Learn Rate & Execution Time 
        self.learn=random.uniform(0,1)
        self.time=random.randint(0,500)

        # set the button trigger
        self.button_train.clicked.connect(self.trainFile)
        self.button_test.clicked.connect(self.testFile)

        # set the combobox trigger
        self.combo.activated.connect(self.setFile)
        
        # set the layout
        layout = QtWidgets.QVBoxLayout()
        layout_upper = QtWidgets.QHBoxLayout() #the upper level layout
        layout_down = QtWidgets.QHBoxLayout() #the lower level layout

        # insert the figure to upper layout
        layout_upper.addWidget(self.canvas_1)
        layout_upper.addWidget(self.canvas_2)

        # insert the label, test editor, button, combobox and table to lower layout
        form_layout = QFormLayout()
        form_layout.addRow(self.label_file,self.combo)
        form_layout.addRow(self.label_learn,self.editor_learn)
        form_layout.addRow(self.label_time,self.editor_time)
        form_layout.addRow(self.button_train,self.button_test)
        layout_down.addLayout(form_layout,7)
        layout_down.addWidget(self.ltable,15)
        layout_down.addWidget(self.rtable,10)
        
        layout.addLayout(layout_upper,42)
        layout.addLayout(layout_down,10)
 
        self.setLayout(layout)

    def setFile(self):
        self.ax_1.cla()
        self.ax_2.cla()
        # x ,y and out is to store the value of two dimension input and one dimension expect output
        self.x=[]
        self.y=[]
        self.out=[]

        # initial the train and test group
        self.x_train=[]
        self.y_train=[]
        self.x_test=[]
        self.y_test=[]
        self.out_train=[]
        self.out_test=[]

        # record the condition to distinguish the two different group
        self.low=int(100)

        # this is to record the bound of x_axis and y_axis 
        self.x_max=float(-100)
        self.x_min=float(100)
        self.y_max=float(-100)
        self.y_min=float(100)

        # record the number of input
        self.count=int(0)

        # record the train and test group
        self.count_train=int(0)
        self.count_test=int(0)

        # random generate a point that seperate train and test set
        self.start = int(0)
        
        # read the input file
        f=open(self.combo.currentText())
        while 1:
            line=f.readline()
            if line=="":
                break
            
            line=line[:len(line)].strip().split(" ")

            # determine the x axis range
            temp=float(line[0])
            if temp < self.x_min :
                self.x_min = temp
            if temp > self.x_max :
                self.x_max = temp
            self.x.append(temp)

            # determine the y axis range
            temp=float(line[1])
            if temp < self.y_min :
                self.y_min = temp
            if temp > self.y_max :
                self.y_max = temp
            self.y.append(temp)
            
            temp=int(line[2])
            if temp < self.low :
                self.low = temp
            self.out.append(temp)

            self.count+=1

        # normalize the out value
        for i in range(self.count) :
            if self.out[i]==self.low :
                self.out[i]=int(0)
            else :
                self.out[i]=int(1)

        # determine whether or not to use 2/3 and 1/3 to divide the train & test group
        if self.count > 10 :
            # calculate the number of train & test group
            self.count_train = int(self.count * 2/3)
            self.count_test = self.count - self.count_train

            # determine the divide point
            self.start = random.randint(0, self.count-1)

            # push the data into train and test set
            for i in range(self.count_test-1) :
                self.x_train.append(self.x[(3*i+self.start)%self.count])
                self.y_train.append(self.y[(3*i+self.start)%self.count])
                self.out_train.append(self.out[(3*i+self.start)%self.count])
                self.x_train.append(self.x[(3*i+self.start+1)%self.count])
                self.y_train.append(self.y[(3*i+self.start+1)%self.count])
                self.out_train.append(self.out[(3*i+self.start+1)%self.count])
                self.x_test.append(self.x[(3*i+self.start+2)%self.count])
                self.y_test.append(self.y[(3*i+self.start+2)%self.count])
                self.out_test.append(self.out[(3*i+self.start+2)%self.count])
            if self.count_test*3 != self.count :
                if self.count_test*3 - self.count == 2 :
                    self.x_test.append(self.x[(self.start-1)%self.count])
                    self.y_test.append(self.y[(self.start-1)%self.count])
                    self.out_test.append(self.out[(self.start-1)%self.count])
                else :
                    self.x_test.append(self.x[(self.start-2)%self.count])
                    self.y_test.append(self.y[(self.start-2)%self.count])
                    self.out_test.append(self.out[(self.start-2)%self.count])
                    self.x_train.append(self.x[(self.start-1)%self.count])
                    self.y_train.append(self.y[(self.start-1)%self.count])
                    self.out_train.append(self.out[(self.start-1)%self.count])
            else :
                self.x_train.append(self.x[(self.start-2)%self.count])
                self.y_train.append(self.y[(self.start-2)%self.count])
                self.out_train.append(self.out[(self.start-2)%self.count])
                self.x_train.append(self.x[(self.start-1)%self.count])
                self.y_train.append(self.y[(self.start-1)%self.count])
                self.out_train.append(self.out[(self.start-1)%self.count])
                self.x_test.append(self.x[(self.start-3)%self.count])
                self.y_test.append(self.y[(self.start-3)%self.count])
                self.out_test.append(self.out[(self.start-3)%self.count])

        else :
            # calculate the number of train & test group
            self.count_train = self.count
            self.count_test = self.count 

            # push the data into train and test set
            for i in range(self.count) :
                self.x_train.append(self.x[i])
                self.y_train.append(self.y[i])
                self.out_train.append(self.out[i])
                self.x_test.append(self.x[i])
                self.y_test.append(self.y[i])
                self.out_test.append(self.out[i])

        # draw the initial left graph
        self.ax_1.set_title("Train Module")
        self.ax_1.set_xlim([self.x_min-1,self.x_max+1])
        self.ax_1.set_ylim([self.y_min-1,self.y_max+1])
        self.ax_1.plot(self.x_train , self.y_train, '.')
        self.canvas_1.draw()

         # draw the initial right graph
        self.ax_2.set_title("Test Module")
        self.ax_2.set_xlim([self.x_min-1,self.x_max+1])
        self.ax_2.set_ylim([self.y_min-1,self.y_max+1])
        self.ax_2.plot(self.x_test , self.y_test, '.')
        self.canvas_2.draw()

        print("Num of Input:",self.count)
        print(self.count_train,self.count_test)

    # define the plotgraph mechanism
    def divideFile(self):
        # this is to store the two different group in train set
        self.x_train_bigger=[]
        self.y_train_bigger=[]
        self.x_train_smaller=[]
        self.y_train_smaller=[]

        # this is to store the two different group in test set
        self.x_test_bigger=[]
        self.y_test_bigger=[]
        self.x_test_smaller=[]
        self.y_test_smaller=[]

        # divive the train group and store them into two set train_bigger & train_smaller
        for i in range(self.count_train):
            if self.z3[i] >= 0.5 : 
                self.x_train_bigger.append(self.z1[i])
                self.y_train_bigger.append(self.z2[i])
            else :
                self.x_train_smaller.append(self.z1[i])
                self.y_train_smaller.append(self.z2[i])

        # divive the test group and store them into two set test_bigger & test_smaller
        # calculate the correct rate
        temp = float(0)
        for i in range(self.count_test):
            temp = math.pow(1 + math.exp(-(self.w3[0]*(-1) + self.w3[1]*math.pow(1 + math.exp(-(self.w1[0]*(-1) + self.w1[1]*self.x_test[i] + self.w1[2]*self.y_test[i])), -1) + self.w3[2]*math.pow(1 + math.exp(-(self.w2[0]*(-1) + self.w2[1]*self.x_test[i] + self.w2[2]*self.y_test[i])), -1))), -1)
            if temp >= 0.5: 
                self.x_test_bigger.append(math.pow(1 + math.exp(-(self.w1[0]*(-1) + self.w1[1]*self.x_test[i] + self.w1[2]*self.y_test[i])), -1))
                self.y_test_bigger.append(math.pow(1 + math.exp(-(self.w2[0]*(-1) + self.w2[1]*self.x_test[i] + self.w2[2]*self.y_test[i])), -1))
                if self.out_test[i]==int(1) :
                    self.correct_test+=1
            else :
                self.x_test_smaller.append(math.pow(1 + math.exp(-(self.w1[0]*(-1) + self.w1[1]*self.x_test[i] + self.w1[2]*self.y_test[i])), -1))
                self.y_test_smaller.append(math.pow(1 + math.exp(-(self.w2[0]*(-1) + self.w2[1]*self.x_test[i] + self.w2[2]*self.y_test[i])), -1))
                if self.out_test[i]==int(0) :
                    self.correct_test+=1
            self.rmse =  self.rmse + math.pow(temp-self.out_test[i], 2)
        self.correct_test /= self.count_test

    # define the plotgraph mechanism
    def trainFile(self):
        # set Learn Rate and Execution Time
        self.learn=float(self.editor_learn.text())
        self.time=int(self.editor_time.text())
        print(self.learn,self.time)

        # set the number of correct distinguishment in train group
        self.correct_train = float(0)

        # set the number of correct distinguishment in test group
        self.correct_test = float(0)

        # set the number of correct distinguishment in test group
        self.rmse = float(0)
        
        line = np.linspace(-5, 5, 50)
        self.w1=[]
        self.w2=[]
        self.w3=[]

        # set the initial key value
        for i in range(3):
            self.w1.append(random.uniform(-1,1))
            self.w2.append(random.uniform(-1,1))
            self.w3.append(random.uniform(-1,1))

        for i in range(self.time):
            index = i % self.count_train
            
            self.y1 = math.pow(1 + math.exp(-(self.w1[0]*(-1) + self.w1[1]*self.x_train[index] + self.w1[2]*self.y_train[index])), -1)
            self.y2 = math.pow(1 + math.exp(-(self.w2[0]*(-1) + self.w2[1]*self.x_train[index] + self.w2[2]*self.y_train[index])), -1)
            self.y3 = math.pow(1 + math.exp(-(self.w3[0]*(-1) + self.w3[1]*self.y1 + self.w3[2]*self.y2)), -1)

            self.theta3 = (self.out_train[index] - self.y3) * self.y3 * (1 - self.y3)
            self.theta1 = self.y1 * (1 - self.y1) * self.theta3 * self.w3[1]
            self.theta2 = self.y2 * (1 - self.y2) * self.theta3 * self.w3[2]

            self.w1[0] = self.w1[0] + self.learn * self.theta1 * (-1)
            self.w1[1] = self.w1[1] + self.learn * self.theta1 * self.x_train[index]
            self.w1[2] = self.w1[2] + self.learn * self.theta1 * self.y_train[index]
            self.w2[0] = self.w2[0] + self.learn * self.theta2 * (-1)
            self.w2[1] = self.w2[1] + self.learn * self.theta2 * self.x_train[index]
            self.w2[2] = self.w2[2] + self.learn * self.theta2 * self.y_train[index]
            self.w3[0] = self.w3[0] + self.learn * self.theta3 * (-1)
            self.w3[1] = self.w3[1] + self.learn * self.theta3 * self.y1
            self.w3[2] =self. w3[2] + self.learn * self.theta3 * self.y2

        self.z1=[]
        self.z2=[]
        self.z3=[]

        # calculate the correct rate
        for i in range(self.count_train) : 
            self.z1.append(math.pow(1 + math.exp(-(self.w1[0]*(-1) + self.w1[1]*self.x_train[i] + self.w1[2]*self.y_train[i])), -1))
            self.z2.append(math.pow(1 + math.exp(-(self.w2[0]*(-1) + self.w2[1]*self.x_train[i] + self.w2[2]*self.y_train[i])), -1))
            self.z3.append(math.pow(1 + math.exp(-(self.w3[0]*(-1) + self.w3[1]*self.z1[i] + self.w3[2]*self.z2[i])), -1))
            if (self.z3[i]<float(0.5) and self.out_train[i]==int(0)) or (self.z3[i]>=float(0.5) and self.out_train[i]==int(1)):
                self.correct_train += 1
            self.rmse = self.rmse + math.pow(self.z3[i]-self.out_train[i], 2)
        self.correct_train /= self.count_train

        # divide the train and test group into different set bigger&smaller
        self.divideFile()
        
        # print the valve & key value on left_table
        self.ltable.setItem(0,1,QTableWidgetItem(str(self.w1[0])))
        self.ltable.setItem(0,2,QTableWidgetItem(str(self.w1[1])))
        self.ltable.setItem(0,3,QTableWidgetItem(str(self.w1[2])))
        self.ltable.setItem(1,1,QTableWidgetItem(str(self.w2[0])))
        self.ltable.setItem(1,2,QTableWidgetItem(str(self.w2[1])))
        self.ltable.setItem(1,3,QTableWidgetItem(str(self.w2[2])))
        self.ltable.setItem(2,1,QTableWidgetItem(str(self.w3[0])))
        self.ltable.setItem(2,2,QTableWidgetItem(str(self.w3[1])))
        self.ltable.setItem(2,3,QTableWidgetItem(str(self.w3[2])))
        
        # print the train rate on right_table
        self.rtable.setItem(0,1,QTableWidgetItem(str(self.correct_train)))

        # draw the graph
        self.ax_1.cla()
        self.ax_1.set_title("Train Module")
        self.ax_1.set_xlim([-0.1,1.1])
        self.ax_1.set_ylim([-0.1,1.1])
        self.ax_1.plot(self.x_train_bigger , self.y_train_bigger, 'b.',
                    self.x_train_smaller , self.y_train_smaller, 'y.',
                    line, (self.w3[0] - self.w3[1]*line) / self.w3[2],'r')
        
        self.canvas_1.draw()

    def testFile(self): 
        # print the test rate on right_table
        self.rtable.setItem(1,1,QTableWidgetItem(str(self.correct_test)))

        # print the RMSE on right_table
        self.rmse = math.pow(self.rmse / self.count, 1/2)
        self.rtable.setItem(2,1,QTableWidgetItem(str(self.rmse)))

        # draw the graph
        line = np.linspace(-5, 5, 50)
        self.ax_2.cla()
        self.ax_2.set_title("Test Module")
        self.ax_2.set_xlim([-0.1,1.1])
        self.ax_2.set_ylim([-0.1,1.1])
        self.ax_2.plot(self.x_test_bigger , self.y_test_bigger, 'b.',
                    self.x_test_smaller , self.y_test_smaller, 'y.',
                    line, (self.w3[0] - self.w3[1]*line) / self.w3[2],'r')
        
        self.canvas_2.draw()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main_window = My_Main_window()
    main_window.show()
    app.exec()
