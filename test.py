import numpy as np

from day2hour import Day2hour

lat = np.load('./sample/lat.npy')
long = np.load('./sample/lon.npy')
Tmax = np.load('./sample/tmax20220601.npy')
Tmin = np.load('./sample/tmin20220601.npy')
RH = np.load('./sample/hm20220601.npy')
wind = np.load('./sample/wsa20220601.npy')
rain = np.load('./sample/rain20220601.npy')
sunhour = np.load('./sample/sunshine20220601.npy')

model = Day2hour(
    doy=200, latitude=lat, longitude=long, sunshine=sunhour, tmax=Tmax, tmin=Tmin, rh=RH, wind=wind, rain=rain
)

model.calcHour()

model.hirrad
model.htemp
model.hrh
model.hwind
model.hrain
