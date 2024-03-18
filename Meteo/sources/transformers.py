# pylint: disable=line-too-long

""" transformers module """

################################################################
## Copyright(C) 2024, Charles Theetten, <chalimede@proton.me> ##
################################################################

################################################################################

from sklearn.base   import TransformerMixin

import numpy        as np

################################################################################

# DEFINITION DES CONSTANTES

EMPTY_DATA = "Empty data"

################################################################################

class TrCleanCloud(TransformerMixin):
    """ TrCleanCloud class """

    def __init__(self):
        """ constructor """

    def fit(self, data):
        """ fit method """
        if data is None:
            raise ValueError(EMPTY_DATA)
        return self

    def transform(self, data):
        """ replace some values about cloud variables """
        data.loc[:, "Cloud9am"] = data.loc[:, "Cloud9am"].replace({ 9 : 8 })
        data.loc[:, "Cloud3pm"] = data.loc[:, "Cloud3pm"].replace({ 9 : 8 })
        return data

################################################################################

class TrCleanNaNRainTomorrow(TransformerMixin):
    """ TrCleanNaNRainTomorrow class """

    def __init__(self):
        """ constructor """

    def fit(self, data):
        """ fit method """
        if data is None:
            raise ValueError(EMPTY_DATA)
        return self

    def transform(self, data):
        """ delete nan in RainTomorrow variable """
        data = data.dropna(subset = ["RainTomorrow"])
        return data

################################################################################

class TrCleanNaNRow(TransformerMixin):
    """ TrCleanNaNRow class """

    def __init__(self):
        """ constructor """

    def fit(self, data):
        """ fit method """
        if data is None:
            raise ValueError(EMPTY_DATA)
        return self

    def transform(self, data):
        """ delete row filled with only nan values """
        data = data.dropna(subset = data.columns.difference(["Date", "Location"]), how = "all")
        return data

################################################################################

class TrCleanRainTomorrow(TransformerMixin):
    """ TrCleanRainTomorrow class """

    def __init__(self):
        """ constructor """

    def fit(self, data):
        """ fit method """
        if data is None:
            raise ValueError(EMPTY_DATA)
        return self

    def transform(self, data):
        """ round RainTomorrow variable """
        data["RainTomorrow"] = data["RainTomorrow"].round()
        return data

################################################################################

class TrCleanRowDate(TransformerMixin):
    """ TrCleanRowDate class """

    def __init__(self, date):
        """ constructor """
        self.date = date

    def fit(self, data):
        """ fit method """
        if data is None:
            raise ValueError(EMPTY_DATA)
        return self

    def transform(self, data):
        """ delete rows below date """
        data = data[data["Date"].dt.year >= self.date]
        return data

################################################################################

class TrClimaticClusters(TransformerMixin):
    """ TrClimaticClusters class """

    CLUSTERS = { "Albury"       : 4, "BadgerysCreek" : 6, "Cobar"            : 4, "Ballarat"     : 7,
                 "Newcastle"    : 5, "NorahHead"     : 2, "NorfolkIsland"    : 2, "Mildura"      : 4,
                 "Sydney"       : 5, "SydneyAirport" : 5, "WaggaWagga"       : 4, "Williamtown"  : 5,
                 "Wollongong"   : 5, "Canberra"      : 7, "Tuggeranong"      : 7, "MountGinini"  : 8,
                 "Bendigo"      : 6, "Sale"          : 6, "MelbourneAirport" : 6, "Melbourne"    : 6,
                 "Nhil"         : 4, "Portland"      : 6, "Watsonia"         : 6, "Dartmoor"     : 6,
                 "GoldCoast"    : 2, "Townsville"    : 1, "Adelaide"         : 5, "MountGambier" : 6,
                 "Woomera"      : 4, "Albany"        : 6, "Witchcliffe"      : 5, "PearceRAAF"   : 5,
                 "Perth"        : 5, "SalmonGums"    : 4, "Walpole"          : 6, "Hobart"       : 7,
                 "AliceSprings" : 3, "Darwin"        : 1, "Katherine"        : 1, "Uluru"        : 3,
                 "CoffsHarbour" : 2, "Moree"         : 4, "Penrith"          : 6, "Richmond"     : 6,
                 "Launceston"   : 7, "Brisbane"      : 3, "Cairns"           : 1,
                 "Nuriootpa"    : 6, "PerthAirport"  : 5 }

    def __init__(self):
        """ constructor """

    def fit(self, data):
        """ fit method """
        if data is None:
            raise ValueError(EMPTY_DATA)
        return self

    def transform(self, data):
        """ create cluster column """
        data["cluster"] = data["Location"]
        data["cluster"] = data["cluster"].replace(__class__.CLUSTERS)
        return data

################################################################################

class TrDaysOfMonth(TransformerMixin):
    """ TrDaysOfMonth class """

    def __init__(self):
        """ constructor """

    def fit(self, data):
        """ fit method """
        if data is None:
            raise ValueError(EMPTY_DATA)
        return self

    def transform(self, data):
        """ create day of the month column """
        data["Days"] = data["Date"].dt.day
        return data

################################################################################

class TrDaysOfYear(TransformerMixin):
    """ TrDaysOfYear class """

    def __init__(self):
        """ constructor """

    def fit(self, data):
        """ fit method """
        if data is None:
            raise ValueError(EMPTY_DATA)
        return self

    def transform(self, data):
        """ create day of the year column """
        data["Days"] = data["Date"].dt.dayofyear
        return data

################################################################################

class TrDiscretizeCloud(TransformerMixin):
    """ TrDiscretizeCloud class """

    def __init__(self):
        """ constructor """

    def discretise_cloud(self, x):
        """ discretise cloud value """
        clouds = [ 0, 1, 2, 3, 4, 5, 6, 7, 8 ]

        for i in range(1, len(clouds)):
            if x < clouds[i] - 0.5:
                x = clouds[i - 1]
                break
            if x >= 7.5:
                x = 8
        x = x / 8
        return x

    def fit(self, data):
        """ fit method """
        if data is None:
            raise ValueError(EMPTY_DATA)
        return self

    def transform(self, data):
        """ discretize cloud variables """
        data.loc[:, "Cloud9am"] = data.loc[:, "Cloud9am"].apply(self.discretise_cloud)
        data.loc[:, "Cloud3pm"] = data.loc[:, "Cloud3pm"].apply(self.discretise_cloud)
        return data

################################################################################

class TrDiscretizeRain(TransformerMixin):
    """ TrDiscretizeRain class """

    RAIN    = { "No" : 0, "Yes" : 1 }
    COLUMNS = [ "RainToday", "RainTomorrow" ]

    def __init__(self):
        """ constructor """

    def fit(self, data):
        """ fit method """
        if data is None:
            raise ValueError(EMPTY_DATA)
        return self

    def transform(self, data):
        """ discretize rain values """
        data.loc[:, __class__.COLUMNS] = data.loc[:, __class__.COLUMNS].replace(__class__.RAIN)
        return data

################################################################################

class TrDiscretizeWindDirection(TransformerMixin):
    """ TrDiscretizeWindDirection class """

    POINTS  = { "N" : 4,  "NNE" : 3,   "NE": 2,  "ENE" : 1,
                "E" : 0,  "ESE" : 15 , "SE": 14, "SSE" : 13,
                "S" : 12, "SSW" : 11,  "SW": 10, "WSW" : 9,
                "W" : 8,  "WNW" : 7,   "NW": 6,  "NNW" : 5 }

    COLUMNS = [ "WindGustDir", "WindDir9am", "WindDir3pm" ]

    def __init__(self):
        """ constructor """

    def fit(self, data):
        """ fit method """
        if data is None:
            raise ValueError(EMPTY_DATA)
        return self

    def transform(self, data):
        """ discretize wind direction """
        data.loc[:, __class__.COLUMNS]  = data.loc[:, __class__.COLUMNS].replace(__class__.POINTS)
        data.loc[:, "WindGustDir_sin"]  = data.loc[:, "WindGustDir"].apply(lambda x : np.sin(np.pi * x / 8) if not np.isnan(x) else x)
        data.loc[:, "WindGustDir_cos"]  = data.loc[:, "WindGustDir"].apply(lambda x : np.cos(np.pi * x / 8) if not np.isnan(x) else x)
        data.loc[:, "WindDir9am_sin"]   = data.loc[:, "WindDir9am"].apply(lambda x : np.sin(np.pi * x / 8) if not np.isnan(x) else x)
        data.loc[:, "WindDir9am_cos"]   = data.loc[:, "WindDir9am"].apply(lambda x : np.cos(np.pi * x / 8) if not np.isnan(x) else x)
        data.loc[:, "WindDir3pm_sin"]   = data.loc[:, "WindDir3pm"].apply(lambda x : np.sin(np.pi * x / 8) if not np.isnan(x) else x)
        data.loc[:, "WindDir3pm_cos"]   = data.loc[:, "WindDir3pm"].apply(lambda x : np.cos(np.pi * x / 8) if not np.isnan(x)  else x)
        data = data.drop(__class__.COLUMNS, axis = 1)
        return data

################################################################################

class TrGPS(TransformerMixin):
    """ TrGPS class """

    LONGITUDES = { "Adelaide": 138.5999312, "Albany": 117.8666667, "AliceSprings": 133.8806662975767, "Albury": 146.9135265,
                   "BadgerysCreek": 150.76428973131323, "Ballarat": 143.8605645, "Bendigo": 144.2826718, "Brisbane": 153.0234991,
                   "Cairns": 145.7721854, "Canberra": 149.1012676, "Cobar": 145.8344444, "CoffsHarbour": 153.11247453419182,
                   "Darwin": 130.8410469, "Dartmoor": 141.2714795276723, "GoldCoast": 153.4283333, "Hobart": 147.3281233,
                   "Katherine": 132.2635993, "Launceston": 147.1373496, "Melbourne": 144.9631732, "MelbourneAirport": 144.84350404526143,
                   "Mildura": 142.1503146, "Moree": 149.8407153, "MountGambier": 140.78030074325983, "MountGinini": 148.7769126989756,
                   "Newcastle": 151.781253, "Nhil": 141.8503146, "NorfolkIsland": 150.097284541572, "NorahHead": 151.57859725363036,
                   "Nuriootpa": 138.9939006, "PearceRAAF": 116.02955403288523, "Penrith": 150.743505, "Perth": 115.8605855,
                   "PerthAirport": 115.96776416210514, "Portland": 141.60127842859012, "Richmond": 144.99555765076107, "Sale": 147.1666667,
                   "SalmonGums": 121.62420381660041, "Sydney": 151.2082848, "SydneyAirport": 151.1754488454341, "Townsville": 146.8239537,
                   "Tuggeranong": 149.0921341, "Uluru": 131.03696147470208, "WaggaWagga": 147.3662603334174, "Walpole": 116.72861905798922,
                   "Watsonia": 145.0837808, "Williamtown": 151.8427778, "Witchcliffe": 115.1004768, "Wollongong": 150.89345,
                   "Woomera" : 136.8237607354683 }

    LATITUDES =  { "Adelaide": -34.9285189, "Albany": -35.0217912, "AliceSprings": -23.7000001, "Albury": -36.0737734,
                   "BadgerysCreek": -33.87509064379364, "Ballarat": -37.5623013, "Bendigo": -36.7590183, "Brisbane": -27.4689682,
                   "Cairns": -16.9206657, "Canberra": -35.2975906, "Cobar": -31.4983333, "CoffsHarbour": -30.29585881521828,
                   "Darwin": -12.46044, "Dartmoor": -37.92325341758002, "GoldCoast": -28.0000001, "Hobart": -42.8819444,
                   "Katherine": -14.4646157, "Launceston": -41.4439222, "Melbourne": -37.8142454, "MelbourneAirport": -37.67059103794948,
                   "Mildura": -34.195274, "Moree": -29.4617202, "MountGambier": -37.82855766933099, "MountGinini": -35.529237792251415,
                   "Newcastle": -32.9272888, "Nhil": -35.2417501, "NorfolkIsland": -29.0417601, "NorahHead": -33.28159706486806,
                   "Nuriootpa": -34.4880556, "PearceRAAF": -31.66741940877939, "Penrith": -33.750278, "Perth": -31.9558933,
                   "PerthAirport": -31.938429146355816, "Portland": -38.3608443353396, "Richmond": -37.823956509075266, "Sale": -38.1166667,
                   "SalmonGums": -32.9784660461056, "Sydney": -33.8698439, "SydneyAirport": -33.931013930916876, "Townsville": -19.2583333,
                   "Tuggeranong": -35.4209771, "Uluru": -25.344490,
                   "WaggaWagga": -35.10235293014164, "Walpole": -34.97227617070785, "Watsonia": -37.7333333,
                   "Williamtown": -32.815, "Witchcliffe": -33.9333333, "Wollongong": -34.4278083, "Woomera": -31.4666667 }

    def __init__(self):
        """ constructor """

    def fit(self, data):
        """ fit method """
        if data is None:
            raise ValueError(EMPTY_DATA)
        return self

    def transform(self, data):
        """ create longitude and latitude columns """
        data["Longitude"]   = data["Location"].replace(__class__.LONGITUDES)
        data["Latitude"]    = data["Location"].replace(__class__.LATITUDES)
        data["Longitude"]   = data["Longitude"].astype(np.float64)
        data["Latitude"]    = data["Latitude"].astype(np.float64)
        return data

################################################################################

class TrSubsetNaN(TransformerMixin):
    """ TrSubsetNaN class """

    def __init__(self, n):
        """ constructor """
        self.n = n

    def fit(self, data):
        """ fit method """
        if data is None:
            raise ValueError(EMPTY_DATA)
        return self

    def transform(self, data):
        """ build cities list with least percentage NaN values """
        x_mean_nan = data.groupby(["Location"]).apply(lambda x : x.notna().mean())
        cities_nan = x_mean_nan.mean(axis = 1).sort_values(ascending = False)
        return data[data["Location"].isin(cities_nan.index[0:self.n])]

################################################################################

class TrZonesRain(TransformerMixin):
    """ SubsetNaN class """

    ZONES = { 50   : 1, 100  : 2, 200  : 3,
              300  : 4, 400  : 5, 600  : 6,
              1000 : 7, 1500 : 8, 2000 : 9,
              3000 : 10 }

    def __init__(self):
        """ constructor """

    def fit(self, data):
        """ fit method """
        if data is None:
            raise ValueError(EMPTY_DATA)
        return self

    def transform(self, data):
        """ build zones based on average annual rainfall """
        data["Year"]    = data["Date"].dt.year
        data["Zone"]    = 0
        cities          = data["Location"].unique()
        rain            = data.groupby(["Location", "Year"])["Rainfall"].sum()

        for city in cities:
            year_rainfall = rain.get(city)
            for year, rain_val in year_rainfall.items():
                for rain_limit, zone in __class__.ZONES.items():
                    if rain_val < rain_limit:
                        data.loc[(data["Location"] == city) & (data["Year"] == year), "Zone"] = zone
                        break
        data = data.drop("Year", axis = 1)
        return data

################################################################################
