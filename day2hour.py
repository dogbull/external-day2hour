## module day2hour.py    - 2022.10.07  by KH Moon
## This module is an array-oriented code for converting daily weather data to hourly data. 
## A Constant(scalar) or numpy array(1D, 2D, 3D) can be used as input data.
##
## Iuput : doy, latitude, longitude, sunshine, tmax, tmin, rh, wind, rain, height, pressure
##         doy(day of year)에 해당하는 날짜의 일별 기상을 시간별 기상자료로 변환시켜줌
##         doy는 숫자 입력, refernece height = 2.0m, pressure = 1000 mbar (고정값)
##          latitude, longitude, sunshine, tmax, tmin, rh, wind, rain는 scalar/vector/array 형식 가능
## Output : 20시간의 hourly temperature(htemp), irradiance(hirrad), relative humidity(hrh), 
##          wind(hwind), rain(hrain)를 numpy array 형식으로 표현
##          (ex) htemp[2] => 03 시의 temperature(scalar/vector/array) 

import numpy as np
import random

class Day2hour():

    def __init__(self, doy, latitude, longitude, sunshine, tmax, tmin, rh, wind, rain, 
                 height=2.0, pressure=1000):
        
        # initialization 
        # element : daily weather data(tmax, tmin, rh, wind, rain, sunhr) - scalr or numpy.array
        #           and latitude, logitude - scalar or numpy.array
        self.hr    = np.arange(24)   # hour (0-23)
        self.doy   = doy             # one day - scalar, not array
        self.lat   = latitude        # radians, not degree ( = np.deg2rad(degrees))
        self.long  = longitude       # radians, not degree ( = np.deg2rad(degrees))
        self.sunhr = sunshine        # hours
        self.tmax  = tmax            # C
        self.tmin  = tmin            # C
        self.tavg  = (tmax + tmin) / 2  # C
        self.rh    = rh              # %
        self.wd    = wind            # m/s
        self.rain  = rain            # mm
        self.hgt   = height          # m
        self.press = pressure        # mbars
        
        # results (24-hour data of all weather elements) by numpy array
        self.hirrad  = None          # W/m2
        self.htemp   = None          # C
        self.hrh     = None          # %
        self.hwind   = None          # m/s
        self.hrain   = None          # mm

    def calcHour(self):
        self.htemp  = self.calcHrTemp()
        self.hrh    = self.calcHrRH()
        self.hwind  = self.calcHrWind()
        self.hrain  = self.calcHrRain()
        self.hirrad = self.calcCloud()

    def solarCalc(self):
        doy = self.doy; lat   = self.lat
        decl  = -0.4093 * np.cos(2 * np.pi * (doy + 10) /365)     # sun declination (rad)
        dA    = np.sin(lat) * np.sin(decl)                     
        dB    = np.cos(lat) * np.cos(decl)
        dayLength = 24 * np.arccos(-dA / dB) / np.pi
        sunSet    = 12 * np.arccos(-dA / dB) / np.pi + 12
        sunRise   = 24 - sunSet
        return (dayLength, sunRise, sunSet)

    def hourAngle(self):
        hr  = self.hr;  doy = self.doy;  long = self.long;
        hr1 = np.tile(hr, long.shape +(1,))                      # hour array fit to longitude array
        B   = 2 * np.pi * (doy - 81) / 364
        EoT = 9.87 * np.sin(2 * B) - 7.53 * np.cos(B) - 1.5 * np.sin(B)
        gammaSM = (long / (np.pi/12)).astype(np.int32) * (np.pi/12)
        corr    = (gammaSM - long) / (np.pi/12)
        corr    = corr.reshape(corr.shape + (1,))
        tcorr   = hr1 + corr + EoT / 60
        hourAngle = np.pi * (tcorr - 12)/12
        return hourAngle

    def calcTemp(self):
        hr = self.hr; Tmax = self.tmax; Tmin = self.tmin
        dayLength, sunRise, sunSet = self.solarCalc()
        sunRise = sunRise.reshape(sunRise.shape + (1,))
        Tavg = np.array((Tmax + Tmin) / 2)
        Tdif = np.array((Tmax - Tmin) / 2)
        Tavg = Tavg.reshape(Tavg.shape + (1,))
        Tdif = Tdif.reshape(Tdif.shape + (1,))

        dTau1 = np.cos(np.pi * (hr + 10) / (sunRise + 10))
        dTau2 = np.cos(np.pi * (hr - sunRise) / (14 - sunRise))
        dTau3 = np.cos(np.pi * (hr - 14) / (10 + sunRise))
        choice = [dTau1, -dTau2, dTau3]
        cond = [hr < sunRise, (hr >= sunRise) & (hr <= 14), hr > 14]
        dTau = np.select(cond, choice)
        hrsTemp = Tavg + Tdif * dTau
        return hrsTemp
    
    def calcHrTemp(self):
        hrsTemp = self.calcTemp()
        result = hrsTemp
        # the order of result data transpose by (24) hours
        if result.ndim   == 1:    ind = (0,);
        elif result.ndim == 2:    ind = (1,0);
        elif result.ndim == 3:    ind = (2,0,1);
        else: print("dimension of input data in over 3 dims")  
        hTemp  = np.transpose(result,ind)
        return hTemp

    def calcHrRH(self):
        rh    = self.rh;  Tmin = self.tmin
        rhcor = np.minimum(0.0148 * rh - 0.1669, 1)
        rhcor = rhcor.reshape(rhcor.shape + (1,))
        temp = self.calcTemp()
        svp = 6.1078 * np.exp(17.269 * temp / (temp + 237.3))
        svpTmin = 6.1078 * np.exp(17.269 * Tmin / (Tmin + 237.3))
        svpTmin = svpTmin.reshape(svpTmin.shape + (1,))
        hourRH  = svpTmin / svp * 100 * rhcor
        result = hourRH
        # the order of result data transpose by (24) hours
        if result.ndim   == 1:    ind = (0,);
        elif result.ndim == 2:    ind = (1,0);
        elif result.ndim == 3:    ind = (2,0,1);
        else: print("dimension of input data in over 3 dims")  
        hourRH  = np.transpose(result,ind)
        return hourRH

    def calcHrWind(self):
        wind = self.wd
        wd  = np.maximum(wind, 0.01)
        wd  = wd.reshape(wd.shape + (1,))
        hrwd = np.tile(np.zeros(24),wd.shape) + wd
        stdRatio = -0.175 * np.log(hrwd) + 0.6151 
        std   = wd * stdRatio / 2
        hourWind = hrwd + np.random.normal(0, std)
        hourWind = np.maximum(hourWind, 0.01)
        result = hourWind
        # the order of result data transpose by (24) hours
        if result.ndim   == 1:    ind = (0,);
        elif result.ndim == 2:    ind = (1,0);
        elif result.ndim == 3:    ind = (2,0,1);
        else: print("dimension of input data in over 3 dims")  
        hourWind  = np.transpose(result,ind)
        return hourWind 

    def calcHrRain(self):
        rain  = self.rain; doy = self.doy;
        rain_ = np.array(rain)
        raininit = np.zeros(24)
        avhrs = np.array(-0.0004 * doy ** 2 + 0.149 * doy - 0.5987)  #연중 일 강우시 발생시간 빈도, 최소값=3
        avhrs = np.maximum(avhrs, 3).astype(int)              
        idx   = np.random.choice(range(24), avhrs, replace=False)    # 24시간 중 배분
        g_gamma = np.random.gamma(1,1,avhrs)                         # 강우 시 강우량 분포
        g_gammaRatio = g_gamma/np.sum(g_gamma)                       # 강우 시 강우량 분포 비율
        raininit[idx] = g_gammaRatio                                 # 시간별 배분
        rain_ = rain_.reshape(rain_.shape +(1,))
        hourRain = rain_ * raininit.T                                # 시간강우량 배분
        result = hourRain
        # the order of result data transpose by (24) hours
        if result.ndim   == 1:    ind = (0,);
        elif result.ndim == 2:    ind = (1,0);
        elif result.ndim == 3:    ind = (2,0,1);
        else: print("dimension of input data in over 3 dims")  
        hourRain  = np.transpose(result,ind)
        return hourRain

    # calculate hourly radiation (using MRM model)
    # seasonal extra solar radiation W m-2, corrected by dist. from E and S(eccen), EoT and standard meridian
    # optical air mass
    def calcIexOam(self): 
        doy   = self.doy;  lat = self.lat;  press = self.press
        decl  = -0.4093 * np.cos(2 * np.pi * (doy + 10) /365)
        gamma = 2.0 * np.pi * (doy - 1) / 365
        Iex   = 1366.1 * (1.00011 + 0.034221 * np.cos(gamma) + 0.00128 * np.sin(gamma) 
                        + 0.000719 * np.cos(2 * gamma) + 0.000077 * np.sin(2 * gamma))
        tau   = self.hourAngle()
        A     = np.sin(decl) * np.sin(lat)
        B     = np.cos(decl) * np.cos(lat)
        A     = A.reshape(A.shape +(1,))
        B     = B.reshape(B.shape +(1,))
        B     = np.cos(tau) * B
        sinbeta = np.maximum(A + B, 0)
        extraSolar = Iex * sinbeta        # W/m2
        theta   = np.pi / 2 - np.arcsin(sinbeta)
        theta   = np.clip(theta, 0, np.pi)
        sza     = np.degrees(theta)               # solar zenith angle(degrees)
        C       = np.power((96.07995 - sza), -1.6364)
        optAirMass  = 1 / (sinbeta + 0.50572 * C)
        corrAirMass = optAirMass * press / 1013.25     # Po = 1013.25 mbar or 101.325 kPa
        return (extraSolar, optAirMass, corrAirMass)

    ### Calculation radiation inhibitors
    #  Tw (H2O vapor), Tmg(mixed gas), Toz(ozone), Tray(Rayleigh scatter), Taero (aerosol)
    def calcT(self):
        Tmax  = self.tmax;  Tmin  = self.tmin;  RH = self.rh; 
        height = self.hgt;   lat  = self.lat
        Tmean   = (Tmax + Tmin) / 2
        Tl     = (Tmean + 273.15) / 100
        es     = np.exp(22.329699 - 49.140396 / Tl - 10.921853 / Tl / Tl - 0.39015156 * Tl)
        em     = es * RH / 100
        uwater = 0.493 * em / (Tmean + 273.15)
        uwater = np.array(uwater)
        uwater = uwater.reshape(uwater.shape +(1,))
    #     uwater = np.reshape(uwater.size, 1)
        Iex,m,mpr = self.calcIexOam()           # Iex (extra solar), m (optical air mass)
        B      = np.maximum(1 + 119.3 * uwater * m, 0.1)
        A      = np.power(B, 0.644)
        Twater = 1 - 3.014 * m * uwater / (A + 5.814 * m * uwater)    # water coefficient

        Tco2 = 1 - 0.721 * mpr * 350 / ((1 + 377.89 * mpr * 350) ** (0.5855) + 3.1709 * mpr * 350)
        Tco = 1 - 0.0062 * mpr * 0.075 / ((1 + 243.67 * mpr * 0.075) ** (0.4246) + 1.7222 * mpr * 0.075)
        Tn2o = 1 - 0.0326 * mpr * 0.28 / ((1 + 107.413 * mpr * 0.28) ** (0.5501) + 0.9093 * mpr * 0.28)
        Tch4 = 1 - 0.0192 * mpr * 1.6 / ((1 + 166.095 * mpr * 1.6) ** (0.4221) + 0.7186 * mpr * 1.6)
        To2 = 1 - 0.0003 * mpr * 209500 / ((1 + 476.934 * mpr * 209500) ** (0.4892) + 0.1261 * mpr * 209500)       
        Tmix = Tco2 * Tco * Tn2o * Tch4 * To2                          # mixed gases coeff.

        Tozone = 1 - 0.2554 * m * 0.3 / ((1 + 6017.26 * m * 0.3) ** (0.204) + 0.471 * m * 0.3)  # from Psiloglou(2007)

        AA = -0.1128 * mpr ** (0.8346)
        BB = 0.9341 - mpr ** (0.9868) + 0.9391 * mpr
        Tray  = np.exp(AA * BB)

        beta = (0.025 + 0.1 * np.cos(lat)) * np.exp(-0.7 * height / 1000) + 0.04      #from Psiloglou(2007)
        beta = np.array(beta)
        beta = beta.reshape(beta.shape +(1,))    

        Taero = np.exp(-1 * m * beta * (0.6777 + 0.1464 * m * beta - 0.00626 * (m * beta) ** 2) ** (-1.3))
        Taero[Taero < 0] = 0.001
        Taeab = 1 - 0.1 * (1 - m + m ** (1.06)) * (1 - Taero)
        Taeab[Taeab < 0] = 0.001
        Tratio = Taero / Taeab     
        Tratio[Tratio < 0] = 0.001
        return (Twater, Tmix, Tozone, Tray, Taero, Taeab, Tratio)

    ### calculate radiations on Clear condition (W/m2) 
    #  Itotal, Ibeam, Idiffuse
    def calcClear(self):
        lat  = self.lat;  height = self.hgt;
        Iex, m, mpr = self.calcIexOam()
        Tw, Tmg, Toz, Tr, Ta, Taeab, Tratio = self.calcT() 
        Ibeam = Iex * Tw * Tr * Toz * Tmg * Ta
        Ibeam = np.maximum(Ibeam, 0.0)

        # Idfsin, clear diffuse by single scattering of direct beam
        dayLength, sunRise, sunSet = self.solarCalc()
        Idfsin  = Iex * Tw * Toz * Tmg * Taeab * 0.5 * (1 - Tratio * Tr)
    #   # diffuse by single scattering of direct beam radiation
        # Idfmul, clear diffuse by multiple scattering of direct beam from Psiloglou(2007)
        beta = (0.025 + 0.1 * np.cos(lat)) * np.exp(-0.7 * height / 1000) + 0.04 
        Ta166 = np.exp(-1 * 1.66 * beta * (0.6777 + 0.1464 * 1.66 * beta - 0.00626 * (1.66 * beta)**2)**(-1.3))
        Ta166 = Ta166.reshape(Ta166.shape +(1,))
        alphag = 0.2
        alphar = 0.0685
        alphaa = 0.16 * (1 - Ta166)
        Idfmul = (Ibeam + Idfsin) * (alphag * (alphar + alphaa)) / (1 - alphag * (alphar + alphaa))
    #   # diffuse by multiple scattering of direct beam radiation
        Idiffuse = Idfsin + Idfmul   # diffuse total at clear condition
        Itotal = Ibeam + Idiffuse
        return (Itotal, Ibeam, Idiffuse, Idfsin, Idfmul)

    ### calculati radiations on Cloud condition (W/m2)
    #  Ictotal, Icbeam, Icdiffuse
    def calcCloud(self):
        sunhour = self.sunhr;  lat = self.lat;  height = self.hgt;
        dayLength, sunRise, sunSet = self.solarCalc()

        Tcloud = 0.75 * sunhour / dayLength     # 0.9 is empirical coeff Psiloglou(2007), 0.75 from Ideriah(1981)
        Tcloud = Tcloud.reshape(Tcloud.shape +(1,))
        Itotal, Ibeam, Idiffuse, Idfsin, Idfmul = self.calcClear()
        Icbeam  = Ibeam * Tcloud                                               # beam radiation at cloud condition
        Icdfsin = Idfsin * Tcloud + 0.33 * (1 - Tcloud) * (Ibeam + Idfsin)     # diffuse single scattering
        beta = (0.025 + 0.1 * np.cos(lat)) * np.exp(-0.7 * height / 1000) + 0.04      # Psiloglou(2007)
        Ta166 = np.exp(-1 * 1.66 * beta * (0.6777 + 0.1464 * 1.66 * beta - 0.00626 * (1.66 * beta)**2)**(-1.3))
        Ta166 = Ta166.reshape(Ta166.shape +(1,))
        alphag = 0.2
        alphar = 0.0685
        alphaa = 0.16 * (1 - Ta166)
        alphac = 0.4 * sunhour / dayLength   # from Psiloglou(2007)
        alphac = alphac.reshape(alphac.shape +(1,))
        Icdfmul = (Icbeam + Icdfsin)*alphag*(alphar + alphaa + alphac)/(1-alphag*(alphar+alphaa+alphac))
        Icdiffuse = Icdfsin + Icdfmul
        Ictotal   = Icbeam + Icdiffuse
        result = Ictotal
        # the order of result data transpose by (24) hours
        if result.ndim   == 1:    ind = (0,);
        elif result.ndim == 2:    ind = (1,0);
        elif result.ndim == 3:    ind = (2,0,1);
        else: print("dimension of input data in over 3 dims")  
        Ictotal  = np.transpose(result,ind)
        return Ictotal

######################################################
#### test input data for "Day2hour" class    
######################################################
    
# lat = np.deg2rad(33)
# long = np.deg2rad(127)
# Tmax = 25
# Tmin = 12
# RH = 80
# wind = 1.0
# rain = 100
# sunhour = 6

# # # doy = np.array([200, 100])
# lat = np.array(np.deg2rad([33,35,37]))
# long = np.array(np.deg2rad([127,128,128]))
# Tmax = np.array([25, 30, 10])
# Tmin = np.array([12, 10, -2])
# RH = np.array([80, 70, 60])
# wind = np.array([1.0, 2.0, 3.0])
# rain = np.array([10, 5, 1])
# sunhour = np.array([6,8,5])

# lat = np.array(np.deg2rad([[33,35],[33,36]]))
# long = np.array(np.deg2rad([[127,128],[127,129]]))
# Tmax = np.array([[25, 30],[25,30]])
# Tmin = np.array([[12, 10],[12,10]])
# RH = np.array([[80, 70],[80,70]])
# wind = np.array([[1.0, 0.0],[1.0,0.0]])
# rain = np.array([[10, 5],[10,5]])
# sunhour = np.array([[6,8],[6,9]])

# lat = np.load('lat.npy')
# long = np.load('lon.npy')
# Tmax = np.load('tmax20220601.npy')
# Tmin = np.load('tmin20220601.npy')
# RH = np.load('hm20220601.npy')
# wind = np.load('wsa20220601.npy')
# rain = np.load('rain20220601.npy')
# sunhour = np.load('sunshine20220601.npy')

######################################################
### test code for  "Day2hour" class

# import day2hour as dh

# model = dh.Day2hour(doy=200, latitude=lat, longitude=long, sunshine=sunhour, tmax=Tmax,
#                  tmin=Tmin, rh=RH, wind=wind, rain=rain)
# model.calcHour()

# model.hirrad
# model.htemp[12]
# model.hrh
# model.hwind
# model.hrain[8]
