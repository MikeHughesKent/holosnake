# -*- coding: utf-8 -*-
"""
Kent-CAS: Camera Acquisition System
Kent-CAS-GUI : Camera Acquisition System GUI

Thread class for image processing of inline holographic microscopy images .

This is a sub-class of ImageProcessorThread provided with Kent-CAS.

@author: Mike Hughes, Applied Optics Group, University of Kent

"""

import sys
import numpy as np
import time

from cas_gui.threads.image_processor_class import ImageProcessorClass

import pyholoscope as pyh

import matplotlib.pyplot as plt



class InlineHoloProcessor(ImageProcessorClass):
    
    mask = None
    crop = None
    filterSize = None
    calibration = None
    refocus = False
    preProcessFrame = None
    autoFocusFlag = False
    invert = False
    showPhase = False
    roi = None
    unwrap = False
    removeTilt = False
    DIC = False
    
    def __init__(self):
        
        super().__init__()
        self.holo = pyh.Holo(pyh.INLINE, 1, 1, cropCentre = (20,20), cropRadius = 10, relativePhase = True)
        
                
    def process(self, inputFrame):
        """ This is called by parent class whenever a frame needs to be processed.
        """
        self.preProcessFrame = inputFrame

        if inputFrame is None:
            return None

        # If we are in inline mode and refocus is false we need to hack this 
        # to return the input images as PyHoloscope has not option to do
        # inline holography without refocusing
        if self.holo.mode == pyh.INLINE and not self.refocus:
            return inputFrame

        outputFrame = self.holo.process(inputFrame)


        if outputFrame is not None:
            if self.showPhase is False:
                outputFrame = pyh.amplitude(outputFrame)
                if self.invert is True:
                    outputFrame = np.max(outputFrame) - outputFrame
            else:
                outputFrame = pyh.phase(outputFrame) 
                if self.unwrap:
                    print("unwrap")
                    outputFrame = pyh.phase_unwrap(outputFrame)
                if self.removeTilt and self.tiltMap is not None:
                    if np.shape(self.tiltMap) == np.shape(outputFrame):
                        outputFrame = outputFrame - self.tiltMap
                if self.DIC:
                    outputFrame = pyh.synthetic_DIC(outputFrame)
            return outputFrame
        
        return inputFrame

   
    def set_depth(self, depth):
        self.holo.set_depth(depth)

    def auto_focus(self, **kwargs):
        
        if self.preProcessFrame is not None:
            return self.holo.auto_focus(self.preProcessFrame.astype('float32'), **kwargs)

