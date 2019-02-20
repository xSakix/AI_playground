from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence
import matplotlib.pyplot as plt


def get_some_data(cols_to_use):
    data = pd.read_csv('data/house_pricing_snapshot/melb_data.csv')
    y = data.Price
    X = data[cols_to_use]
    my_imputer = Imputer()
    imputed_X = my_imputer.fit_transform(X)
    return imputed_X, y


# cols_to_use = ['Suburb', 'Address', 'Rooms', 'Type', 'Price', 'Method', 'SellerG',
#                'Date', 'Distance', 'Postcode', 'Bedroom2', 'Bathroom', 'Car',
#                'Landsize', 'BuildingArea', 'YearBuilt', 'CouncilArea', 'Lattitude',
#                'Longtitude', 'Regionname', 'Propertycount']

# I think the price could be related to Rooms(more rooms bigger price), YearBuilt(newer bigger price),,
# and post code (the price should vary base don post code)

cols_to_use = ['Rooms', 'YearBuilt', 'Postcode']
# get_some_data is defined in hidden cell above.
X, y = get_some_data(cols_to_use)
# scikit-learn originally implemented partial dependence plots only for Gradient Boosting models
# this was due to an implementation detail, and a future release will support all model types.
my_model = GradientBoostingRegressor()
# fit the model as usual
my_model.fit(X, y)
print(my_model.feature_importances_)
# Here we make the plot
my_plots = plot_partial_dependence(my_model,
                                   features=[0, 2],  # column numbers of plots we want to show
                                   X=X,  # raw predictors data.
                                   feature_names=cols_to_use,  # labels on graphs
                                   grid_resolution=10)  # number of values to plot on x axis
plt.show()
