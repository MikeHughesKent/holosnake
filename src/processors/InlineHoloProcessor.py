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

from ImageProcessorThread import ImageProcessorThread

import pyholoscope as pyh



class InlineHoloProcessor(ImageProcessorThread):
    
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
    
    def __init__(self, inBufferSize, outBufferSize, **kwargs):
        
        super().__init__(inBufferSize, outBufferSize, **kwargs)
        self.holo = pyh.Holo(pyh.INLINE, 1, 1)
        
                
    def process_frame(self, inputFrame):
        """ This is called by parent class whenever a frame needs to be processed.
        """
        self.preProcessFrame = inputFrame

        if self.refocus == True and inputFrame is not None:
            outputFrame = self.holo.process(inputFrame)

            if outputFrame is not None:
                if self.showPhase is False:
                    outputFrame = pyh.amplitude(outputFrame)
                    if self.invert is True:
                        outputFrame = np.max(outputFrame) - outputFrame
                else:
                    outputFrame = pyh.phase(pyh.relative_phase_self(outputFrame, self.roi)) / (2 * 3.14) * 255
           

            return outputFrame
        return inputFrame

    def handle_flags(self):
       
        if self.autoFocusFlag:
            self.autoFocusFlag = False
            #t1 = time.perf_counter()
            #print(self.holo.auto_focus(self.preProcessFrame, **self.autoFocusFlags))
            #print(time.perf_counter() - t1)
        

    def auto_focus(self, **kwargs):
        
        if self.preProcessFrame is not None:
            return self.holo.auto_focus(self.preProcessFrame.astype('float32'), **kwargs)
            #self.autoFocusFlag = True
            #self.autoFocusFlags = kwargs
        
    def update_settings(self):
        pass