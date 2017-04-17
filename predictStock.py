import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

dates = []
prices = []

# Fill the dates & prices list from the csv
def get_data(fileName):
	with open(fileName, 'r') as csvfile:
		csvFileReader =  csv.reader(csvfile)
		# Skip the column names
		next(csvFileReader)

		for row  in csvFileReader:
			dates.append(int(row[0].split('-')[0]))
			prices.append(float(row[1]))
		return

# Build Support Vector Regression Models
def predict_prices(dates, prices, x):
	dates = np.reshape(dates, (len(dates), 1))

	
	# Linear Model
	svr_lin = SVR(kernel = 'linear', C = 1e3)
	#Polynomial Model
	svr_poly = SVR(kernel = 'poly', C = 1e2, degree = 2)
	# Radial Basisc Function Model
	svr_rbf = SVR(kernel = 'rbf', C = 1e3, gamma = 0.1)

	svr_lin.fit(dates, prices)
	svr_poly.fit(dates, prices)
	svr_rbf.fit(dates, prices)

	plt.scatter(dates, prices, color = 'black', label = 'Data')
	plt.plot(dates, svr_rbf.predict(dates), color = 'red', label = 'RBF Model')
	plt.plot(dates, svr_lin.predict(dates), color = 'green', label = 'Linear Model')
	plt.plot(dates, svr_poly.predict(dates), color = 'blue', label = 'Polynomial Model')
	plt.xlabel('Date')
	plt.ylabel('Price')
	plt.title('Support Vector Regression')
	plt.legend()
	plt.show()

	return svr_rbf.predict(x)[0], svr_lin.predict(x)[0], svr_poly.predict(x)[0]

get_data('googl.csv')
predicted_price = predict_prices(dates, prices, 29)




