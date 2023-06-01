# -*- coding: utf-8 -*-
"""
HoloSnAkE : Inline Holographci Microscopy GUI

@author: Mike Hughes, Applied Optics Group, University of Kent

"""


import sys 
from pathlib import Path

sys.path.append(str(Path('../../pyholoscope/src')))
sys.path.append(str(Path('../../cas/src')))
sys.path.append(str(Path('../../cas/src/widgets')))
sys.path.append(str(Path('../../cas/src/cameras')))
sys.path.append(str(Path('../../cas/src/threads')))
sys.path.append(str(Path('../../cas/src/threads')))
sys.path.append(str(Path('processors')))


import time
import numpy as np
import math
import pickle

from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPalette, QColor, QImage, QPixmap, QPainter, QPen, QGuiApplication
from PyQt5.QtGui import QPainter, QBrush, QPen


from PIL import Image

from CAS_GUI_Base import CAS_GUI


import cv2 as cv

from ImageAcquisitionThread import ImageAcquisitionThread
from image_display import ImageDisplay

import matplotlib.pyplot as plt
#import pyqtgraph as pg
from cam_control_panel import *

from InlineHoloProcessor import InlineHoloProcessor
import logging

import pyholoscope
#logging.disable(logging.CRITICAL)

class HoloGUI(CAS_GUI):
    
    AUTHOR = "AOG"
    APP_NAME = "InlineHoloGUI"
    cuda = False
    if cuda is True:
        try:
            import cupy
            
        except:
            print("CUDA not found.")
            cuda = False
            
    
    def __init__(self,parent=None):
        
        self.sourceFilename = r"..\tests\test_data\usaf1.tif"
        controlPanelSize = 300

        super(HoloGUI, self).__init__(parent)
        
        self.handle_change_show_processing_options(1)

        self.exportStackDialog = ExportStackDialog()
        
     
    
    def create_layout(self):
        
        self.setWindowTitle("Kent Inline Holography")       
        
        self.outerLayout = QVBoxLayout()

        self.layout = QHBoxLayout()
        self.mainDisplayFrame = QVBoxLayout()
        self.mosaicDisplayFrame = QVBoxLayout()
        
        # Create the image display widget which will show the video
        self.mainDisplay = ImageDisplay(name = "mainDisplay")
        self.mainDisplay.isStatusBar = True
        self.mainDisplay.autoScale = True
        self.mainDisplay.setMinimumWidth(500)        
          
        # Add the camera display to a parent layout
        self.mainDisplayFrame.addWidget(self.mainDisplay)
                              
        # Create the panel with main menu and camera control options (e.g. exposure)
        self.camControlPanel = init_cam_control_panel(self, self.controlPanelSize)   
        
        # Add custom buttons to main menu       
        self.mainMenuLayout.addItem(verticalSpacer:= QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum))
        self.mainMenuBackBtn = QPushButton('Acquire Background')
        self.mainMenuLayout.addWidget(self.mainMenuBackBtn)
        self.mainMenuBackBtn.clicked.connect(self.acquire_background_click)
        
        self.mainMenuLoadBackBtn = QPushButton('Load Background')
        self.mainMenuLayout.addWidget(self.mainMenuLoadBackBtn)
        self.mainMenuLoadBackBtn.clicked.connect(self.load_background_from_click)
        
        self.mainMenuSaveBackBtn = QPushButton('Save Background')
        self.mainMenuLayout.addWidget(self.mainMenuSaveBackBtn)
        self.mainMenuSaveBackBtn.clicked.connect(self.save_background_click)
        
        # Create the processing panels
        self.holoPanel = self.init_inline_holo_process_panel(self.controlPanelSize)
        self.refocusPanel = self.init_refocus_panel(self.controlPanelSize)
        
        # Create panel with 'Show processing options' checkbox
        self.visibilityControl = QWidget()
        self.visibilityControl.setLayout(visLayout:= QVBoxLayout())
        self.showProcessingOptionsCheck = QCheckBox("Show Processing Options", objectName = "showProcessingOptionsCheck")
        self.showProcessingOptionsCheck.toggled.connect(self.handle_change_show_processing_options)
        visLayout.addWidget(self.showProcessingOptionsCheck)
        visLayout.addStretch()        
        
        # Assemble the layout
        self.leftControl = QVBoxLayout()
        self.leftControl.addWidget(self.camControlPanel)
        self.leftControl.addWidget(self.refocusPanel)
        
        self.layout.addLayout(self.mainDisplayFrame)
        self.layout.addLayout(self.leftControl)
        self.layout.addWidget(self.holoPanel)
        self.leftControl.addWidget(self.visibilityControl)

        widget = QWidget()
        widget.setLayout(self.outerLayout)
        
        self.outerLayout.addLayout(self.layout)
        self.logobar = QHBoxLayout()
        
        kentlogo = QLabel()
        pixmap = QPixmap('../res/kent_logo_2.png')
        kentlogo.setPixmap(pixmap) #.scaled(200, 405, QtCore.Qt.KeepAspectRatio))
        
        self.logobar.addWidget(kentlogo)
        
        self.outerLayout.addLayout(self.logobar)

        # Set the central widget of the Window. Widget will expand
        # to take up all the space in the window by default.
        self.setCentralWidget(widget)
       
        self.setWindowIcon(QtGui.QIcon('../res/icon.png'))

        # Set the central widget of the Window. Widget will expand
        # to take up all the space in the window by default.
        self.setCentralWidget(widget)
        
        
    def create_processors(self):
        if self.imageThread is not None:
            inputQueue = self.imageThread.get_image_queue()
        else:
            inputQueue = None
        
        if self.imageProcessor is None:
            self.imageProcessor = InlineHoloProcessor(10,10, inputQueue = inputQueue)
            if self.imageProcessor is not None:
                self.imageProcessor.start()
        
        #self.imageProcessor = BundleProcessorHandler(10,10, inputQueue = self.imageThread.get_image_queue())
        self.handle_changed_processing()
        
        
    def init_refocus_panel(self, panelSize):
        
        panel = QWidget()
        panel.setLayout(topLayout:=QVBoxLayout())
        panel.setMaximumWidth(panelSize)
        panel.setMinimumWidth(panelSize)
        groupBox = QGroupBox("Refocusing")
        topLayout.addWidget(groupBox)
        groupBox.setLayout(layout:=QVBoxLayout())
       
        self.holoDepthSlider = QSlider(QtCore.Qt.Horizontal, objectName = 'holoDepthSlider')
        self.holoDepthSlider.setTickPosition(QSlider.TicksBelow)
        self.holoDepthSlider.setTickInterval(10)
        self.holoDepthSlider.setMaximum(5000)
        self.holoDepthSlider.valueChanged[int].connect(self.handle_depth_slider)       
        
        self.holoDepthInput = QDoubleSpinBox(objectName='holoDepthInput')
        self.holoDepthInput.setKeyboardTracking(False)
        self.holoDepthInput.setMaximum(10**6)
        self.holoDepthInput.setMinimum(-10**6)
       
        self.holoAutoFocusBtn=QPushButton('Auto Depth')
        
        self.holoCreateStackBtn = QPushButton('Export Depth Stack')
        
        layout.addWidget(QLabel('Refocus depth (microns):'))
        layout.addWidget(self.holoDepthSlider)
        layout.addWidget(self.holoDepthInput)
        
        layout.addWidget(self.holoAutoFocusBtn)
        layout.addWidget(self.holoCreateStackBtn)

        self.holoAutoFocusBtn.clicked.connect(self.auto_focus_click)
        self.holoDepthInput.valueChanged[float].connect(self.handle_changed_processing)
        self.holoCreateStackBtn.clicked.connect(self.create_depth_stack_click)

        topLayout.addStretch()
        
        return panel          
    

    def init_inline_holo_process_panel(self, panelSize):        
        
        self.holoWavelengthInput = QDoubleSpinBox(objectName='holoWavelengthInput')
        self.holoWavelengthInput.setMaximum(10**6)
        self.holoWavelengthInput.setMinimum(-10**6)
        
        self.holoPixelSizeInput = QDoubleSpinBox(objectName='holoPixelSizeInput')
        self.holoPixelSizeInput.setMaximum(10**6)
        self.holoPixelSizeInput.setMinimum(-10**6)
        
            
        self.holoRefocusCheck = QCheckBox("Refocus", objectName='holoRefocusCheck')
        self.holoBackgroundCheck = QCheckBox("Background", objectName='holoBackgroundCheck')
        self.holoNormaliseCheck = QCheckBox("Normalise", objectName='holoNormaliseheck')

        self.holoWindowCombo = QComboBox(objectName='holoWindowCombo')
        self.holoWindowCombo.addItems(['None', 'Circular', 'Rectangular'])

        self.holoWindowThicknessInput = QDoubleSpinBox(objectName='holoWindowThicknessInput')
        self.holoWindowThicknessInput.setMaximum(10**6)
        self.holoWindowThicknessInput.setMinimum(-10**6)
        
        self.holoAutoFocusMinInput = QDoubleSpinBox(objectName='holoAutoFocusMinInput')
        self.holoAutoFocusMinInput.setMaximum(10**6)
        self.holoAutoFocusMinInput.setMinimum(-10**6)
        
        self.holoAutoFocusMaxInput = QDoubleSpinBox(objectName='holoAutoFocusManInput')
        self.holoAutoFocusMaxInput.setMaximum(10**6)
        self.holoAutoFocusMaxInput.setMinimum(-10**6)        
        
        self.holoAutoFocusCoarseDivisionsInput = QDoubleSpinBox(objectName='holoAutoFocusCoarseDivisionsInput')
        self.holoAutoFocusCoarseDivisionsInput.setMaximum(10**6)
        self.holoAutoFocusCoarseDivisionsInput.setMinimum(0)
        
        self.holoAutoFocusROIMarginInput = QDoubleSpinBox(objectName='holoAutoFocusROIMarginInput')
        self.holoAutoFocusROIMarginInput.setMaximum(10**6)
        self.holoAutoFocusROIMarginInput.setMinimum(0)
        
        self.holoSliderMaxInput = QSpinBox(objectName='holoSliderMaxInput')
        self.holoSliderMaxInput.setMaximum(10**6)
        self.holoSliderMaxInput.setMinimum(0)
        self.holoSliderMaxInput.setKeyboardTracking(False)
      
        self.holoDownsampleInput = QSpinBox(objectName='holoDownsampleInput')
        self.holoDownsampleInput.setMaximum(10)
        self.holoDownsampleInput.setMinimum(1)
        self.holoDownsampleInput.setKeyboardTracking(False)        
        
        holoPanel = QWidget()
        holoPanel.setLayout(topLayout:=QVBoxLayout())
        holoPanel.setMaximumWidth(panelSize)
        holoPanel.setMinimumWidth(panelSize)        
        
        groupBox = QGroupBox("Inline Holography")
        topLayout.addWidget(groupBox)
        groupBox.setLayout(layout:=QVBoxLayout())
        
        layout.addWidget(QLabel('Wavelegnth (microns):'))
        layout.addWidget(self.holoWavelengthInput)
        layout.addWidget(QLabel('Pixel Size (microns):'))
        layout.addWidget(self.holoPixelSizeInput)       
        
        layout.addWidget(self.holoRefocusCheck)
        layout.addWidget(self.holoBackgroundCheck)
        layout.addWidget(self.holoNormaliseCheck)

         
        layout.addWidget(QLabel('Window:'))
        layout.addWidget(self.holoWindowCombo)

        layout.addWidget(QLabel("Window Thickness (px):"))
        layout.addWidget(self.holoWindowThicknessInput) 
        
        layout.addWidget(QLabel("Autofocus Min (microns):"))
        layout.addWidget(self.holoAutoFocusMinInput) 
        
        layout.addWidget(QLabel("Autofocus Max (microns):"))
        layout.addWidget(self.holoAutoFocusMaxInput)
        
        layout.addWidget(QLabel("Autofocus Coarse Intervals:"))
        layout.addWidget(self.holoAutoFocusCoarseDivisionsInput)
        
        layout.addWidget(QLabel("Autofocus ROI Margin (px):"))
        layout.addWidget(self.holoAutoFocusROIMarginInput)
        
        layout.addWidget(QLabel("Depth Slider Max (microns):"))
        layout.addWidget(self.holoSliderMaxInput)

        layout.addWidget(QLabel("Downsample Factor:"))
        layout.addWidget(self.holoDownsampleInput)
        layout.addStretch()
        
        self.holoWavelengthInput.valueChanged[float].connect(self.handle_changed_processing)
        self.holoPixelSizeInput.valueChanged[float].connect(self.handle_changed_processing)
        self.holoRefocusCheck.stateChanged.connect(self.handle_changed_processing)
        self.holoBackgroundCheck.stateChanged.connect(self.handle_changed_processing)
        self.holoNormaliseCheck.stateChanged.connect(self.handle_changed_processing)

        self.holoWindowThicknessInput.valueChanged[float].connect(self.handle_changed_processing)
        self.holoWindowCombo.currentIndexChanged[int].connect(self.handle_changed_processing)
        self.holoSliderMaxInput.valueChanged[int].connect(self.handle_changed_processing)
        self.holoDownsampleInput.valueChanged[int].connect(self.handle_changed_processing)

        return holoPanel    


    def init_inline_holo_sr_panel(self, panelSize):
        
        self.srReferencedBtn=QPushButton('Set Reference Frame')
        self.srShowOffsetCheck = QCheckBox('Show SR Offset', objectName = 'srShowOffsetCheck')
        self.srAngleInput = QSpinBox(objectName = 'seAngleInput')
        
        srPanel = QWidget()
        srPanel.setLayout(topLayout:=QVBoxLayout())
        srPanel.setMaximumWidth(panelSize)
        srPanel.setMinimumWidth(panelSize)        
        
        groupBox = QGroupBox("Super Resolution")
        groupBox.setLayout(layout:=QVBoxLayout())       
    
        layout.addWidget(self.srReferencedBtn)                    

        self.srXOffsetLabel = QLabel()
        self.srYOffsetLabel = QLabel()
        statusPanel = QWidget()
        statusPanel.setLayout(statusLayout:=QGridLayout())
        statusPanel.setMaximumWidth(panelSize)
        statusPanel.setMinimumWidth(panelSize)

        statusLayout.addWidget(QLabel('X Offset:'),1,0)
        statusLayout.addWidget(QLabel('Y Offset:'),2,0)
        
        statusLayout.addWidget(self.srXOffsetLabel,1,1)
        statusLayout.addWidget(self.srYOffsetLabel,2,1)
        
        layout.addWidget(self.srShowOffsetCheck)
        layout.addWidget(QLabel('Rot Angle (deg):'))
        layout.addWidget(self.srAngleInput)
        layout.addWidget(statusPanel)
        topLayout.addWidget(groupBox)

        layout.addStretch()
      
        self.srReferencedBtn.clicked.connect(self.sr_reference_click)
        
        return srPanel        
    

    def update_image_display(self):
       
        if self.currentProcessedImage is not None:
            self.mainDisplay.set_mono_image(self.currentProcessedImage)
        else:
            if self.currentImage is not None:
                self.mainDisplay.set_mono_image(self.currentImage)
                   
            
    def sr_reference_click(self):
        self.srReferenceImage = self.currentProcessedImage         
            
    
    def handle_changed_processing(self):   
        """ When changes are made to processing options, set up the image
        processor to process the images as required.        
        """
        

        # Match depth slider to depth numeric input        
        self.holoDepthSlider.setValue(int(self.holoDepthInput.value()))

        # The max value of the slider is controlled by a numeric
        self.holoDepthSlider.setMaximum(int(self.holoSliderMaxInput.value()))


        # Everything else is only possible if we have an image processor
        if self.imageProcessor is not None:
            
            self.imageProcessor.holo.set_use_cuda(self.cuda) 
            
            self.imageProcessor.holo.set_downsample(self.holoDownsampleInput.value())
            
            if self.holoBackgroundCheck.isChecked() and self.backgroundImage is not None:
                self.imageProcessor.holo.set_background(self.backgroundImage)
            else:
                self.imageProcessor.holo.set_background(None)
                
            if self.holoNormaliseCheck.isChecked() and self.backgroundImage is not None:
                self.imageProcessor.holo.set_normalise(self.backgroundImage)
            else:
                self.imageProcessor.holo.set_normalise(None)
                
            
            # Remaining options are only relevant if we refocus    
            if self.holoRefocusCheck.isChecked():
                
                self.imageProcessor.refocus = True
                
                if self.holoWavelengthInput.value() != self.imageProcessor.holo.wavelength / 10**6:
                    self.imageProcessor.holo.set_wavelength(self.holoWavelengthInput.value()/ 10**6)
                
                targetPixelSize = self.holoPixelSizeInput.value() / 10**6                   
                if targetPixelSize != self.imageProcessor.holo.pixelSize:
                    self.imageProcessor.holo.set_pixel_size(targetPixelSize)
                    
                if self.holoDepthInput.value() != self.imageProcessor.holo.depth / 10**6:
                    self.imageProcessor.holo.set_depth(self.holoDepthInput.value()/ 10**6)
                
                if self.holoWindowCombo.currentText() == "Circular":
                    self.imageProcessor.holo.set_auto_window(True)
                    self.imageProcessor.holo.set_window_shape('circle')
                    self.imageProcessor.holo.set_window_thickness(self.holoWindowThicknessInput.value())
                elif self.holoWindowCombo.currentText() == "Rectangular":
                    self.imageProcessor.holo.set_auto_window(True)
                    self.imageProcessor.holo.set_window_shape('square')
                    self.imageProcessor.holo.set_window_thickness(self.holoWindowThicknessInput.value())
                else:
                    self.imageProcessor.holo.clear_window()
                    self.imageProcessor.holo.set_auto_window(False)
                                                        
            else:
                self.imageProcessor.refocus = False
        
        self.update_file_processing()
        
   
    def handle_change_show_processing_options(self, event):
        if self.showProcessingOptionsCheck.isChecked():
            self.holoPanel.show()
        else:  
            self.holoPanel.hide()
            
    
    def auto_focus_click(self):
     
        if self.mainDisplay.roi is not None:
            roi = PyHoloscope.roi(self.mainDisplay.roi[0], self.mainDisplay.roi[1], self.mainDisplay.roi[2] - self.mainDisplay.roi[0], self.mainDisplay.roi[3] - self.mainDisplay.roi[1])
        else:
            roi = None
        autofocusMax = self.holoAutoFocusMaxInput.value() / 1000
        autofocusMin = self.holoAutoFocusMinInput.value() / 1000
        numSearchDivisions = int(self.holoAutoFocusCoarseDivisionsInput.value())
        autofocusROIMargin = self.holoAutoFocusROIMarginInput.value()
        if self.imageThread is not None:
            self.imageThread.pause()
        autoFocus = (self.imageProcessor.auto_focus(roi = roi, margin = autofocusROIMargin, depthRange = (autofocusMin, autofocusMax), coarseSearchInterval = numSearchDivisions))
        self.holoDepthInput.setValue(autoFocus * 1000) 

        if self.imageThread is not None:
            self.imageThread.resume()
            

    def handle_depth_slider(self):
        self.holoDepthInput.setValue(int(self.holoDepthSlider.value()))

     
    def apply_default_settings(self):
        pass
        
        
    def create_depth_stack_click(self):
        """ Creates a depth stack over a specified range.
        """
        
        if self.imageProcessor is not None and self.imageProcessor.preProcessFrame is not None:
            if self.exportStackDialog.exec():
                try:
                    filename = QFileDialog.getSaveFileName(self, 'Select filename to save to:', '', filter='*.tif')[0]
                except:
                    filename = None
                if filename is not None and filename != '':
                     depthRange = (self.exportStackDialog.depthStackMinDepthInput.value() / 1000, self.exportStackDialog.depthStackMaxDepthInput.value() / 1000)
                     nDepths = int(self.exportStackDialog.depthStackNumDepthsInput.value())
                     QApplication.setOverrideCursor(Qt.WaitCursor)
                     depthStack = self.imageProcessor.holo.depth_stack(self.imageProcessor.preProcessFrame, depthRange, nDepths)
                     QApplication.restoreOverrideCursor()
                     depthStack.write_intensity_to_tif(filename)
        else:
              QMessageBox.about(self, "Error", "A hologram is required to create a depth stack.")  
        
        
class ExportStackDialog(QDialog):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Export Stack")
        self.setMinimumWidth(300)

        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel

        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.layout = QVBoxLayout()
        self.depthStackMinDepthInput = QDoubleSpinBox()
        self.depthStackMinDepthInput.setMaximum(10**6)
        self.depthStackMinDepthInput.setValue(0)
        self.depthStackMaxDepthInput = QDoubleSpinBox()
        self.depthStackMaxDepthInput.setMaximum(10**6)
        self.depthStackMaxDepthInput.setValue(1)

        self.depthStackNumDepthsInput = QSpinBox()
        self.depthStackNumDepthsInput.setMaximum(10**6)
        self.depthStackNumDepthsInput.setValue(10)

        self.layout.addWidget(QLabel("Start Depth (mm):"))
        self.layout.addWidget(self.depthStackMinDepthInput)
        self.layout.addWidget(QLabel("End Depth (mm):"))
        self.layout.addWidget(self.depthStackMaxDepthInput)
        self.layout.addWidget(QLabel("Number of Depths:"))
        self.layout.addWidget(self.depthStackNumDepthsInput)
        
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)
        
    
if __name__ == '__main__':    
    app=QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # Now use a palette to switch to dark colors:
    # Now use a palette to switch to dark colors:
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.Window, QColor(0, 0, 0))

    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(55, 45, 45))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, Qt.black)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.Button, QColor(63, 63, 63))

    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(palette)
       
    window=HoloGUI()
    window.show()
    sys.exit(app.exec_())

