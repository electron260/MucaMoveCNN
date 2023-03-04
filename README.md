# Muca Move

To realize a multi-touch surface, we will use a Muca board.
This board has been designed by Marc Teyssier and enables multi-touch sensing with projected mutual capacitance.
You can get all information about how to make a multi-touch surface [here](https://dvic.devinci.fr/tutorial/how-to-make-a-multitouch-surface/edit).

## Getting Started

### Arduino upload 

In the esp32muca, you will find the code main.cpp to upload on the micro-controler, and the Muca librairy needed for the upload.  

On line 38 and 39 of the main.cpp code to upload, there are two tables with values corresponding to the Tx and Rx pins used by the Muca board.
1 when the pin is used and 0 when it's not. It is therefore necessary to adapt these values to the number of electrodes used.  
```c++
  bool rx[]={1,1,1,1,1,1,1,1,1,1,1,0};
  bool tx[]={1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0};
```
### Display

PROPLOT.py is the python code that displays the movements on the surface. 
When the program is executed and the display has started, it is possible to make some adjustments. Indeed, buttons are available to manually calibrate certain details. You can change the threshold silence if you do not want to display the noise present on your surface. We can also modify the gain, accordingly modifying the amplification of the values sent from the Muca card or even modifying the maximum range of the values received by Muca.


## Authors

  - **Hugo Daumain** 
    [github](https://github.com/electron260)

  - **Nicolas Stas** 
    [github](https://github.com/COLVERTYETY)
    

