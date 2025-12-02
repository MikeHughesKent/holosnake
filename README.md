# HoloSnake - A holographic microscopy GUI in Python

HoloSnake is a graphical user interface for holographic microscopy written in Python. It supports both inline and off-axis holography,
and can be used for real-time reconstruction as well as analysis of saved holograms.

It uses the [PyHoloscope](https://www.github.com/mikehugheskent/pyholoscope) package to process raw holograms and the [CAS](https://www.github.com/mikehugheskent/cas) for the GUI.

It is currently under development with minimal documentation, but is fully functional.

## Installation

Pip install will be available soon, but for now it is necessary to do a manual install. 

Download the HoloSnake source as a ZIP from GitHub. You will also need to download [PyHoloscope](https://www.github.com/mikehugheskent/pyholoscope) and [CAS](https://www.github.com/mikehugheskent/cas). Alternatively, CAS can be pip installed using ``pip install cas-gui``.

The ``PyHoloscope`` and ``HoloSnake`` folders must be in the same parent directory. (The CAS folder must also be in the same directory if downloading rather than pip installing). The PyHoloscope and HoloSnake folders should have those exact names, note that when downloading from GitHub they may have a 'main' attached to the name which should be removed.

A list of dependences is in the ``requirements.txt`` file in the ``holoscope`` folder. You can run ``pip install -r requirements.txt`` to ensure you have all requirements. If you would like to use specific camera inputs you may need additional dependencies, see the [CAS](https://www.github.com/mikehugheskent/cas) documentation for details.

## Getting Started

Launch the GUI by running ``holosnake.py`` in the ``holosnake/src`` folder.

```bash
python holosnake.py
```

### Inline Holography

In the GUI, select 'Image Source' on the left hand menu and in the 'Camera Source' drop-down menu select 'File'. Press the 'Load File' button and select the example file in the examples folder ``inline_example_holo.tif``. A hologram showing out of focus paramecium will appear.

As this is an example image for inline holography, open the 'Off Axis Holography' menu on the left hand side and then ensure that 'Off Axis Demodulation' is toggled off.

Go to the 'Settings' Menu, and enter a wavelength of 0.63 microns and a pixel size of 1 micron. Drag the slider on the right hand side to adjust the focus. The paramecium should be in focus at approximately 12900 microns.

To load the background, go to 'Settings' and click 'Load Background' and select ``inline_example_back.tif``. Subtraction of the background image is toggled using the 'Background (Inline)' option at the top of the Settings panel. The same background image can be used to normalise (flat-field) the hologram by toggling 'Normalise'. Toggle 'Invert' to invert the image.

### Off-Axis Holography

As above, load in a hologram file, this time selecting ``off_axis_example_holo.tif``. Go to the 'Off Axis Holography' menu, and click 'Calibrate Off Axis' to determine the modulation frequency. Enabled demodulation by toggling 'Off Axis Demodulation' on the the same menu. Since this example image is in focus, either move the focus slider to zero, or disable refocusing by toggling 'Refocus' in the 'Focus' menu.

A background hologram can be loaded as for inline holography, by selecting 'Load Background' in the Settings Menu. The 'Background (Inline)' option has no effect for Off-Axis Holography, but the background can be used for flat-fielding by toggling 'Normalise'.

To view the phase image, toggle 'Show Phase' on the Settings Menu. On the Phase Menu, toggle 'Relative Phase' to subtract the phase from the background image.

### Live Camera Images

To perform holographic reconstruction on live images, select a camera source in the 'Image Source' Menu and then select 'Live Imaging' on the Main Menu.






