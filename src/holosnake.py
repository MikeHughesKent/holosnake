# -*- coding: utf-8 -*-
"""
HoloSnake : Inline Holographic Microscopy GUI

This GUI is built on CAS-GUI which contains most of the GUI functionality. This
file sets up the specific aspects of the GUI needed for inline holography.
Holographic reconsutruction, refocusing etc. is performed using the PyHoloscope
package.

@author: Mike Hughes, Applied Optics Group, University of Kent

"""

import sys 
from pathlib import Path

# Paths to CAS and PyHoloscope
sys.path.append(str(Path('../../cas/src')))
sys.path.append(str(Path('../../pyholoscope/src')))

import time
import numpy as np
import math
import pickle
import matplotlib.pyplot as plt
import os

from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon, QPalette, QColor, QImage, QPixmap, QPainter, QPen, QGuiApplication
from PyQt5.QtGui import QPainter, QBrush, QPen

from PIL import Image
import cv2 as cv

from cas_gui.base import CAS_GUI
from processors.inline_holo_processor import InlineHoloProcessor
import pyholoscope

resPath = "..\\..\\cas\\res"


class HoloGUI(CAS_GUI):
    
    AUTHOR = "AOG"
    APP_NAME = "InlineHoloGUI"
    windowTitle = "HoloSnake"
    logoFilename = '../res/kent_logo_3.png'
    resPath = "..\\..\\cas\\res"
    processor = InlineHoloProcessor
    cuda = True
    multiCore = True
    sharedMemory = True
    sharedMemoryArraySize = (2048,2048)
    rawImageBufferSize = 5
    
    if cuda is True:
        try:
            import cupy
            
        except:
            print("CUDA not found, defaulting to CPU")
            cuda = False
    
    
    def __init__(self,parent=None):
        
        self.sourceFilename = r"..\tests\test_data\usaf1.tif"

        super(HoloGUI, self).__init__(parent)
        self.exportStackDialog = ExportStackDialog()
        
     
    
    def create_layout(self):
        """ Overrides default CAS-GUI layout to add additional controls.
        """
        
        super().create_layout()  
        
        # Create the additional menu buttons needed 
        self.calibrationMenuButton = self.create_menu_button("Calibration", QIcon(os.path.join(self.resPath, 'icons', 'grid_white.svg')), self.calibration_menu_button_clicked, True, True, 6)
        self.phaseMenuButton = self.create_menu_button("Phase", QIcon(os.path.join(self.resPath, 'icons', 'grid_white.svg')), self.phase_menu_button_clicked, True, True, 6)
        self.autofocusMenuButton = self.create_menu_button("Auto Focus", QIcon(os.path.join(self.resPath, 'icons', 'grid_white.svg')), self.autofocus_menu_button_clicked, True, True, 6)

        self.stackButton = self.create_menu_button("Depth Stack", QIcon('../res/icons/copy_white.svg'), self.depth_stack_clicked, False, False, 5)

        # Create the additional menu panels needed for HoloBundle
        self.calibrationPanel = self.create_calibration_panel()
        self.phasePanel = self.create_phase_panel()
        self.autofocusPanel = self.create_autofocus_panel()

               
        # Create the long depth slider
        self.create_focus_panel()
        
        
    
    def create_calibration_panel(self):
        """ Create the panel with calibration options"""   
        
        widget, layout = self.panel_helper(title = "Calibration")    
        
        self.mainMenuBackBtn = QPushButton('Acquire Background')
        layout.addWidget(self.mainMenuBackBtn)
        self.mainMenuBackBtn.clicked.connect(self.acquire_background_clicked)
        
        self.mainMenuLoadBackBtn = QPushButton('Load Background')
        layout.addWidget(self.mainMenuLoadBackBtn)
        self.mainMenuLoadBackBtn.clicked.connect(self.load_background_from_clicked)
        
        self.mainMenuSaveBackBtn = QPushButton('Save Background')
        layout.addWidget(self.mainMenuSaveBackBtn)
        self.mainMenuSaveBackBtn.clicked.connect(self.save_background_clicked)
        
        self.holoCalibrateOffAxisBtn = QPushButton('Calibrate Off Axis')
        layout.addWidget(self.holoCalibrateOffAxisBtn)
        self.holoCalibrateOffAxisBtn.clicked.connect(self.calibrate_off_axis_clicked)
        
        layout.addStretch()
        
        return widget
    
    def create_autofocus_panel(self):
        """ Create the panel with calibration options"""   
        
        widget, layout = self.panel_helper(title = "Auto Focus")  
        
        self.holoAutoFocusMinInput = QDoubleSpinBox(objectName='holoAutoFocusMinInput')
        self.holoAutoFocusMinInput.setMaximum(10**6)
        self.holoAutoFocusMinInput.setMinimum(-10**6)
        
        self.holoAutoFocusMaxInput = QDoubleSpinBox(objectName='holoAutoFocusMaxInput')
        self.holoAutoFocusMaxInput.setMaximum(10**6)
        self.holoAutoFocusMaxInput.setMinimum(-10**6)        
        
        self.holoAutoFocusCoarseDivisionsInput = QDoubleSpinBox(objectName='holoAutoFocusCoarseDivisionsInput')
        self.holoAutoFocusCoarseDivisionsInput.setMaximum(10**6)
        self.holoAutoFocusCoarseDivisionsInput.setMinimum(0)
        
        self.holoAutoFocusROIMarginInput = QDoubleSpinBox(objectName='holoAutoFocusROIMarginInput')
        self.holoAutoFocusROIMarginInput.setMaximum(10**6)
        self.holoAutoFocusROIMarginInput.setMinimum(0)
        
        layout.addWidget(QLabel("Autofocus Min (microns):"))
        layout.addWidget(self.holoAutoFocusMinInput) 
        
        layout.addWidget(QLabel("Autofocus Max (microns):"))
        layout.addWidget(self.holoAutoFocusMaxInput)
        
        layout.addWidget(QLabel("Autofocus Coarse Intervals:"))
        layout.addWidget(self.holoAutoFocusCoarseDivisionsInput)
        
        layout.addWidget(QLabel("Autofocus ROI Margin (px):"))
        layout.addWidget(self.holoAutoFocusROIMarginInput)
        
        layout.addStretch()
        
        return widget
    
    def create_phase_panel(self):
        """ Create the panel with calibration options"""   
        
        widget, layout = self.panel_helper(title = "Phase") 
        
        self.holoRelativePhaseCheck = QCheckBox("Relative Phase")
        layout.addWidget(self.holoRelativePhaseCheck)
        self.holoRelativePhaseCheck.stateChanged.connect(self.processing_options_changed)
        
        self.holoUnWrapPhaseCheck = QCheckBox("Unwrap Phase")
        layout.addWidget(self.holoUnWrapPhaseCheck)
        self.holoUnWrapPhaseCheck.stateChanged.connect(self.processing_options_changed)
        
        self.holoRemoveTiltCheck = QCheckBox("Remove Tilt")
        layout.addWidget(self.holoRemoveTiltCheck)
        self.holoRemoveTiltCheck.stateChanged.connect(self.processing_options_changed)
        
        self.holoDetectTiltBtn = QPushButton("Detect Tilt")
        layout.addWidget(self.holoDetectTiltBtn)
        self.holoUnWrapPhaseCheck.clicked.connect(self.detect_tilt_clicked)
        
        self.holoDICCheck = QCheckBox("Synthetic DIC")
        layout.addWidget(self.holoDICCheck)
        self.holoDICCheck.stateChanged.connect(self.processing_options_changed)
        
       
        layout.addStretch()
        
        return widget
    


    def create_focus_panel(self): 
    
        """ Create a long slider for focusing
        """
        self.holoDepthInput = QDoubleSpinBox(objectName='holoDepthInput')
        self.holoDepthInput.setKeyboardTracking(False)
        self.holoDepthInput.setMaximum(10**6)
        self.holoDepthInput.setMinimum(-10**6)
        self.holoDepthInput.setSingleStep(10)
        self.holoDepthInput.setMinimumWidth(90)
        self.holoDepthInput.setMaximumWidth(90)
        self.longFocusWidget = QWidget(objectName = "long_focus")
        self.longFocusWidget.setContentsMargins(0,0,0,0)
        self.longFocusWidgetLayout = QVBoxLayout()
        self.longFocusWidget.setMinimumWidth(190)
        self.longFocusWidget.setMaximumWidth(190)
        self.longFocusWidget.setLayout(self.longFocusWidgetLayout)
        self.holoLongDepthSlider = QSlider(QtCore.Qt.Vertical, objectName = 'longHoloDepthSlider')
        self.holoLongDepthSlider.setInvertedAppearance(True)
        self.refocusTitle = QLabel("Refocus")
        self.longFocusWidgetLayout.addWidget(self.refocusTitle, alignment=QtCore.Qt.AlignHCenter)
        self.refocusTitle.setProperty("subheader", "true")
        self.refocusTitle.setStyleSheet("QLabel{padding:5px}")
        self.holoLongDepthSlider.setStyleSheet("QSlider{padding:20px}")
        self.longFocusWidgetLayout.addWidget( self.holoLongDepthSlider, alignment=QtCore.Qt.AlignHCenter)
        self.contentLayout.addWidget(self.longFocusWidget) 
        self.longFocusWidgetLayout.addWidget(QLabel('Depth, \u03bcm'),alignment=QtCore.Qt.AlignHCenter)
        self.longFocusWidgetLayout.addWidget(self.holoDepthInput,alignment=QtCore.Qt.AlignHCenter)  
        self.longFocusWidget.setStyleSheet("QWidget{padding:0px; margin:0px;background-color:rgba(30, 30, 60, 255)}")
        self.holoDepthInput.valueChanged[float].connect(self.focus_depth_changed)
        self.holoDepthInput.setStyleSheet("QDoubleSpinBox{padding: 5px; background-color: rgba(255, 255, 255, 255); color: black; font-size:9pt}")
        self.holoLongDepthSlider.valueChanged[int].connect(self.long_depth_slider_changed)       
        self.holoLongDepthSlider.setTickPosition(QSlider.TicksBelow)
        self.holoLongDepthSlider.setTickInterval(100)
        self.holoLongDepthSlider.setMaximum(5000)
        
        # Stylesheets for Focus Panel
        file = "../res/holosnake_focus_slider.css"        
        with open(file,"r") as fh:
            self.holoLongDepthSlider.setStyleSheet(fh.read())     
          
            

    def add_settings(self, layout):     
        """ Adds Holography options to Settings Panel.
        """
        
        self.holoOffAxisCheck = QCheckBox("Off Axis Demodulation", objectName='holoOffAxisCheck')
       
        
        self.holoWavelengthInput = QDoubleSpinBox(objectName='holoWavelengthInput')
        self.holoWavelengthInput.setMaximum(10**6)
        self.holoWavelengthInput.setMinimum(-10**6)
        
        self.holoPixelSizeInput = QDoubleSpinBox(objectName='holoPixelSizeInput')
        self.holoPixelSizeInput.setMaximum(10**6)
        self.holoPixelSizeInput.setMinimum(-10**6)        
            
        self.holoRefocusCheck = QCheckBox("Refocus", objectName='holoRefocusCheck')
        self.holoBackgroundCheck = QCheckBox("Background", objectName='holoBackgroundCheck')
        self.holoNormaliseCheck = QCheckBox("Normalise", objectName='holoNormaliseCheck')
        self.holoInvertCheck = QCheckBox("Invert", objectName='holoInvertCheck')
        self.holoShowPhaseCheck = QCheckBox("Show Phase", objectName='holoShowPhaseCheck')

        self.holoWindowCombo = QComboBox(objectName='holoWindowCombo')
        self.holoWindowCombo.addItems(['None', 'Circular', 'Rectangular'])

        self.holoWindowThicknessInput = QDoubleSpinBox(objectName='holoWindowThicknessInput')
        self.holoWindowThicknessInput.setMaximum(10**6)
        self.holoWindowThicknessInput.setMinimum(-10**6)
        
        
        
        self.holoSliderMaxInput = QDoubleSpinBox(objectName='holoSliderMaxInput')        
        self.holoSliderMaxInput.setMaximum(10**6)
        self.holoSliderMaxInput.setMinimum(0)
        self.holoSliderMaxInput.setKeyboardTracking(False)
        
        self.holoSliderMinInput = QDoubleSpinBox(objectName='holoSliderMinInput')        
        self.holoSliderMinInput.setMaximum(0)
        self.holoSliderMinInput.setMinimum(-10**6)
        self.holoSliderMinInput.setKeyboardTracking(False)
      
        self.holoDownsampleInput = QSpinBox(objectName='holoDownsampleInput')
        self.holoDownsampleInput.setMaximum(10)
        self.holoDownsampleInput.setMinimum(1)
        self.holoDownsampleInput.setKeyboardTracking(False)     
        
        self.holoOffAxisCentreX = QSpinBox(objectName='holoOffAxisCentreX')
        self.holoOffAxisCentreY = QSpinBox(objectName='holoOffAxisCentreY')
        self.holoOffAxisRadiusX = QSpinBox(objectName='holoOffAxisRadiusX')
        self.holoOffAxisRadiusY = QSpinBox(objectName='holoOffAxisRadiusY ')
        
        self.holoOffAxisCentreX.setMaximum(100000)
        self.holoOffAxisCentreY.setMaximum(100000)
        self.holoOffAxisRadiusX.setMaximum(100000)
        self.holoOffAxisRadiusY.setMaximum(100000)
        
        self.holoOffAxisCentreX.setMinimum(1)
        self.holoOffAxisCentreY.setMinimum(1)
        self.holoOffAxisRadiusX.setMinimum(1)
        self.holoOffAxisRadiusY.setMinimum(1)
        
        layout.addWidget(self.holoOffAxisCheck)
        layout.addWidget(self.holoRefocusCheck)

        
        
        layout.addWidget(QLabel('Wavelegnth (microns):'))
        layout.addWidget(self.holoWavelengthInput)
        
        layout.addWidget(QLabel('Pixel Size (microns):'))
        layout.addWidget(self.holoPixelSizeInput)       
        
        layout.addWidget(self.holoBackgroundCheck)
        layout.addWidget(self.holoNormaliseCheck)
        layout.addWidget(self.holoInvertCheck)
        layout.addWidget(self.holoShowPhaseCheck)
         
        layout.addWidget(QLabel('Window:'))
        layout.addWidget(self.holoWindowCombo)

        layout.addWidget(QLabel("Window Thickness (px):"))
        layout.addWidget(self.holoWindowThicknessInput) 
        
                
        layout.addWidget(QLabel("Off Axis Modulation Freq (X) (px):"))
        layout.addWidget(self.holoOffAxisCentreX)
        
        layout.addWidget(QLabel("Off Axis Modulation Freq (Y) (px):"))
        layout.addWidget(self.holoOffAxisCentreY)
        
        layout.addWidget(QLabel("Off Axis Crop Radius (X) (px):"))
        layout.addWidget(self.holoOffAxisRadiusX)
        
        layout.addWidget(QLabel("Off Axis Crop Radius (Y) (px):"))
        layout.addWidget(self.holoOffAxisRadiusY)
        
        layout.addWidget(QLabel("Depth Slider Max (microns):"))
        layout.addWidget(self.holoSliderMaxInput)
        
        layout.addWidget(QLabel("Depth Slider Min (microns):"))
        layout.addWidget(self.holoSliderMinInput)

        layout.addWidget(QLabel("Downsample Factor:"))
        layout.addWidget(self.holoDownsampleInput)
        layout.addStretch()
        
        self.holoWavelengthInput.valueChanged[float].connect(self.processing_options_changed)
        self.holoPixelSizeInput.valueChanged[float].connect(self.processing_options_changed)
        self.holoRefocusCheck.stateChanged.connect(self.processing_options_changed)
        self.holoOffAxisCheck.stateChanged.connect(self.processing_options_changed)
        self.holoBackgroundCheck.stateChanged.connect(self.processing_options_changed)
        self.holoNormaliseCheck.stateChanged.connect(self.processing_options_changed)
        self.holoShowPhaseCheck.stateChanged.connect(self.processing_options_changed)
        self.holoInvertCheck.stateChanged.connect(self.processing_options_changed)
        self.holoWindowThicknessInput.valueChanged[float].connect(self.processing_options_changed)
        self.holoWindowCombo.currentIndexChanged[int].connect(self.processing_options_changed)
        self.holoSliderMaxInput.valueChanged[float].connect(self.processing_options_changed)
        self.holoSliderMinInput.valueChanged[float].connect(self.processing_options_changed)
        
        self.holoDownsampleInput.valueChanged[int].connect(self.processing_options_changed)

        self.holoOffAxisCentreX.valueChanged[int].connect(self.processing_options_changed)
        self.holoOffAxisCentreY.valueChanged[int].connect(self.processing_options_changed)
        self.holoOffAxisRadiusX.valueChanged[int].connect(self.processing_options_changed)
        self.holoOffAxisRadiusY.valueChanged[int].connect(self.processing_options_changed)
     
        return 

    def focus_depth_changed(self):
        
        if self.imageProcessor is not None:
        
            self.imageProcessor.pipe_message("set_depth", self.holoDepthInput.value()/ 10**6 )
            self.imageProcessor.get_processor().holo.set_depth(self.holoDepthInput.value()/ 10**6)
        # Match depth slider to depth numeric input        
        self.holoLongDepthSlider.setValue(int(self.holoDepthInput.value()))
        self.update_file_processing()

        
    def processing_options_changed(self):   
        """ When changes are made to processing options, set up the image
        processor to process the images as required.        
        """
        
        # Match depth slider to depth numeric input        
        self.holoLongDepthSlider.setValue(int(self.holoDepthInput.value()))

        # The min/max value of the slider is controlled by a numeric
        self.holoLongDepthSlider.setMaximum(int(self.holoSliderMaxInput.value()))
        self.holoLongDepthSlider.setMinimum(int(self.holoSliderMinInput.value()))
        self.holoLongDepthSlider.setTickInterval(int(np.abs(self.holoSliderMaxInput.value() - self.holoSliderMaxInput.value()) / 100))

        # Everything else is only possible if we have an image processor
        if self.imageProcessor is not None:
            
            self.imageProcessor.get_processor().holo.set_use_cuda(self.cuda) 
            
            self.imageProcessor.get_processor().holo.set_downsample(self.holoDownsampleInput.value())
            
            self.imageProcessor.get_processor().DIC = self.holoDICCheck.isChecked()
            self.imageProcessor.get_processor().removeTilt = self.holoRemoveTiltCheck.isChecked()
            self.imageProcessor.get_processor().unwrap = self.holoUnWrapPhaseCheck.isChecked()

            
            if self.holoBackgroundCheck.isChecked() and self.backgroundImage is not None:
                self.imageProcessor.get_processor().holo.set_background(self.backgroundImage)
            else:
                self.imageProcessor.get_processor().holo.set_background(None)
                
            if self.holoNormaliseCheck.isChecked() and self.backgroundImage is not None:
                self.imageProcessor.get_processor().holo.set_normalise(self.backgroundImage)
            else:
                self.imageProcessor.get_processor().holo.set_normalise(None)
                
            
            if self.holoShowPhaseCheck.isChecked():
                self.imageProcessor.get_processor().showPhase = True
                self.imageProcessor.get_processor().holo.set_relative_phase(self.holoBackgroundCheck.isChecked())
                self.mainDisplay.set_colormap('twilight')
                if self.mainDisplay.roi is not None:
                    self.imageProcessor.get_processor().roi = pyholoscope.Roi(*self.mainDisplay.roi)
                else:
                    self.imageProcessor.get_processor().roi = None
            else:
                self.imageProcessor.get_processor().showPhase = False
                self.imageProcessor.get_processor().invert = self.holoInvertCheck.isChecked()
                self.mainDisplay.set_colormap('gray')

            if self.holoOffAxisCheck.isChecked():
                self.imageProcessor.get_processor().holo.set_mode(pyholoscope.Holo.OFF_AXIS)
                self.imageProcessor.get_processor().holo.set_crop_centre((self.holoOffAxisCentreX.value(),self.holoOffAxisCentreY.value()))
                self.imageProcessor.get_processor().holo.set_crop_radius((self.holoOffAxisRadiusX.value(),self.holoOffAxisRadiusY.value()))

            
            
            else:
                self.imageProcessor.get_processor().holo.set_mode(pyholoscope.Holo.INLINE)
            
            # Remaining options are only relevant if we refocus    
            if self.holoRefocusCheck.isChecked():
                
                self.imageProcessor.get_processor().refocus = True
                self.imageProcessor.get_processor().holo.set_refocus(True)
                
                if self.holoWavelengthInput.value() != self.imageProcessor.get_processor().holo.wavelength / 10**6:
                    if self.holoWavelengthInput.value() > 0:
                        self.imageProcessor.get_processor().holo.set_wavelength(self.holoWavelengthInput.value()/ 10**6)
                
                targetPixelSize = self.holoPixelSizeInput.value() / 10**6                   
                if targetPixelSize != self.imageProcessor.get_processor().holo.pixelSize:
                    if targetPixelSize > 0:
                        self.imageProcessor.get_processor().holo.set_pixel_size(targetPixelSize)
                    
                if self.holoDepthInput.value() != self.imageProcessor.get_processor().holo.depth / 10**6:
                    self.imageProcessor.get_processor().holo.set_depth(self.holoDepthInput.value()/ 10**6)
                
                if self.holoWindowCombo.currentText() == "Circular":
                    self.imageProcessor.get_processor().holo.set_auto_window(True)
                    self.imageProcessor.get_processor().holo.set_window_shape('circle')
                    self.imageProcessor.get_processor().holo.set_window_thickness(self.holoWindowThicknessInput.value())
                elif self.holoWindowCombo.currentText() == "Rectangular":
                    self.imageProcessor.get_processor().holo.set_auto_window(True)
                    self.imageProcessor.get_processor().holo.set_window_shape('square')
                    self.imageProcessor.get_processor().holo.set_window_thickness(self.holoWindowThicknessInput.value())
                else:
                    self.imageProcessor.get_processor().holo.clear_window()
                    self.imageProcessor.get_processor().holo.set_auto_window(False)
                                                        
            else:
                self.imageProcessor.get_processor().refocus = False

            # This is needed if using multicore processing to update the
            # copy of the processor class on the other core
            self.imageProcessor.update_settings()        
        
        # Needed if we are processing a file
        self.update_file_processing()
        
   
    
    def auto_focus_clicked(self):
        """ Handles auto focus click.
        """
     
        if self.mainDisplay.roi is not None:
            roi = pyholoscope.Roi(self.mainDisplay.roi[0], self.mainDisplay.roi[1], self.mainDisplay.roi[2] - self.mainDisplay.roi[0], self.mainDisplay.roi[3] - self.mainDisplay.roi[1])
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
            

    def long_depth_slider_changed(self):
        self.holoDepthInput.setValue(int(self.holoLongDepthSlider.value()))       

     
    def apply_default_settings(self):
        pass
        
      
    def calibration_menu_button_clicked(self):
        self.expanding_menu_clicked(self.calibrationMenuButton, self.calibrationPanel)
    
    def autofocus_menu_button_clicked(self):
        self.expanding_menu_clicked(self.autofocusMenuButton, self.autofocusPanel)
      
    def phase_menu_button_clicked(self):
        self.expanding_menu_clicked(self.phaseMenuButton, self.phasePanel)
           
    
    
    def calibrate_off_axis_clicked(self):
        
        if self.imageProcessor is not None:
            if self.backgroundImage is not None:
                self.imageProcessor.get_processor().holo.calib_off_axis(self.backgroundImage)
            elif self.currentImage is not None:
                self.imageProcessor.get_processor().holo.calib_off_axis(self.currentImage)
            else:
                QMessageBox.about(self, "Error", "Hologram or background image required.")   
                return
            self.holoOffAxisCentreX.setValue(int(self.imageProcessor.get_processor().holo.cropCentre[0]))
            self.holoOffAxisCentreY.setValue(int(self.imageProcessor.get_processor().holo.cropCentre[1]))
            self.holoOffAxisRadiusX.setValue(int(self.imageProcessor.get_processor().holo.cropRadius[0]))
            self.holoOffAxisRadiusY.setValue(int(self.imageProcessor.get_processor().holo.cropRadius[1]))
            self.imageProcessor.update_settings()
            self.update_file_processing()
    
    
    def detect_tilt_clicked(self):
        #pyholoscope.obtain_tilt
        pass
    
    def depth_stack_clicked(self):
        """ Creates a depth stack over a specified range.
        """
        
        if self.imageProcessor is not None and self.currentImage is not None:
            if self.exportStackDialog.exec():
                try:
                    filename = QFileDialog.getSaveFileName(self, 'Select filename to save to:', '', filter='*.tif')[0]
                except:
                    filename = None
                if filename is not None and filename != '':
                     depthRange = (self.exportStackDialog.depthStackMinDepthInput.value() / 1000, self.exportStackDialog.depthStackMaxDepthInput.value() / 1000)
                     nDepths = int(self.exportStackDialog.depthStackNumDepthsInput.value())
                     QApplication.setOverrideCursor(Qt.WaitCursor)
                     depthStack = self.imageProcessor.get_processor().holo.depth_stack(self.currentImage, depthRange, nDepths)
                     QApplication.restoreOverrideCursor()
                     depthStack.write_intensity_to_tif(filename)
        else:
              QMessageBox.about(self, "Error", "A hologram is required to create a depth stack.")  
        
        

class ExportStackDialog(QDialog):
    """ Dialog box that appears when export depth stack is clicked."
    """
    
    def __init__(self):
        super().__init__()
        
        file=os.path.join(resPath, 'cas_modern.css')
        with open(file,"r") as fh:
            self.setStyleSheet(fh.read())

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
       
    window=HoloGUI()
    window.show()
    sys.exit(app.exec_())

