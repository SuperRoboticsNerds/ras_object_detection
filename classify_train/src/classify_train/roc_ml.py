import numpy as np 


def dropData(data, percent, start =0):
	n_rows = data.shape[0]
	rows_drop = int(percent * len(data))
	if start == 0:
		out_data = data[:(n_rows - rows_drop), :]
	else:
		out_data = data[rows_drop:,:]

	return out_data



def accuracy(y_preds, y_actual):
	return float(sum(y_preds == y_actual)) / len(y_preds)