cd models
rm unet.py 2>/dev/null
wget https://raw.githubusercontent.com/ELEKTRONN/elektronn3/f754796d861f1cfe1c19dfc7819087972573ce40/elektronn3/models/unet.py
patch <unet.patch
mv unet.py baseline_UNET3D.py
