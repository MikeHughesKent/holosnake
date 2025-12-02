# -*- coding: utf-8 -*-
"""
Thread class for image processing of holographic microscopy images .

This is a sub-class of ImageProcessorThread provided with Kent-CAS.

"""

import sys
import numpy as np
import time

from cas_gui.threads.image_processor_class import ImageProcessorClass

import pyholoscope as pyh

import matplotlib.pyplot as plt


class HoloProcessor(ImageProcessorClass):

    mask = None
    crop = None
    filterSize = None
    calibration = None
    refocus = False
    preProcessFrame = None
    autoFocusFlag = False
    invert = False
    show_phase = False
    roi = None
    unwrap = False
    tiltMap = None
    removeTilt = False
    DIC = False

    def __init__(self):
        super().__init__()
        self.holo = pyh.Holo(
            pyh.INLINE, 1, 1, crop_centre=(20, 20), crop_radius=10, relative_phase=True
        )


    def process(self, inputFrame):
        """This is called by parent class whenever a frame needs to be processed."""
        self.preProcessFrame = inputFrame

        if inputFrame is None:
            return None

        # If we are in inline mode and refocus is false we need to hack this
        # to return the input images as PyHoloscope has no option to do
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
                    outputFrame = pyh.phase_unwrap(outputFrame)
                if self.removeTilt and self.tiltMap is not None:
                    if np.shape(self.tiltMap) == np.shape(outputFrame):
                        outputFrame = outputFrame - self.tiltMap
                if self.DIC:
                    outputFrame = pyh.synthetic_DIC(outputFrame)
            return outputFrame

        return inputFrame


    def obtain_tilt(self, inputFrame):
        if inputFrame is not None and self.holo is not None:
            phase = pyh.phase_unwrap(pyh.phase(self.holo.process(inputFrame)))
            self.tiltMap = pyh.obtain_tilt(phase)
        else:
            self.tiltMap = None


    def set_depth(self, depth):
        self.holo.set_depth(depth)
        

    def auto_focus(self, **kwargs):
        if self.preProcessFrame is not None:
            return self.holo.auto_focus(
                self.preProcessFrame.astype("float32"), **kwargs
            )
