import numpy, cvxopt
from cvxopt import solvers

def SVM(X, Y, C = 50.0, e = 1.0):
	n_samples = len(X)
	n_inputs = X[0].shape[0]

	P00 = numpy.identity(n_inputs, float)
	P01 = numpy.zeros((n_inputs, n_samples), float)
	P10 = numpy.zeros((n_samples, n_inputs), float)
	P11 = numpy.zeros((n_samples, n_samples), float)
	P0 = numpy.hstack([P00, P01])
	P1 = numpy.hstack([P10, P11])
	P = numpy.vstack([P0, P1])

	Q0 = numpy.zeros((n_inputs, 1), float)
	Q1 = numpy.vstack([C for i in range(n_samples)])
	Q = numpy.vstack([Q0, Q1])

	G00 = numpy.vstack([numpy.transpose(numpy.dot(x, y)) for x, y in zip(X, Y)])
	G01 = numpy.identity(n_samples, float)
	G10 = numpy.zeros((n_samples, n_inputs), float)
	G11 = numpy.identity(n_samples, float)
	G0 = numpy.hstack([G00, G01])
	G1 = numpy.hstack([G10, G11])
	G = numpy.vstack([G0, G1])

	H0 = numpy.vstack([e for i in range(n_samples)])
	H1 = numpy.zeros((n_samples, 1), float)
	H = numpy.vstack([H0, H1])

	svm = solvers.qp(cvxopt.matrix(P), cvxopt.matrix(Q), cvxopt.matrix(-G), cvxopt.matrix(-H))
	return svm

def getDataset(dim = 10, size = 1000, scale = 10):
	ccentre1 = scale * numpy.random.rand(dim, 1)
	ccentre2 = scale * numpy.random.rand(dim, 1)
	X = [ccentre1 + numpy.random.rand(dim, 1) for _ in range(size / 2)] + [ccentre2 + numpy.random.rand(dim, 1) for _ in range(size / 2)]
	Y = [numpy.array([1.0]).reshape(1, 1) for _ in range(size / 2)] + [numpy.array([-1.0]).reshape(1, 1) for _ in range(size / 2)]

	ccentre1[dim - 1][0] = 1.0
	ccentre2[dim - 1][0] = 1.0
	for x in X:
		x[dim - 1][0] = 1.0
	return X, Y, ccentre1, ccentre2

def intersection(point1, point2, plane):
	return (numpy.sum(plane * point2) * point1 - numpy.sum(plane * point1) * point2) / numpy.sum((point2 - point1) * plane)

X, Y, ccentre1, ccentre2 = getDataset()
print ('[CLUSTER1]', ccentre1)
print ('[CLUSTER2]', ccentre2)
print ('[MIDPOINT]', (ccentre1 + ccentre2) / 2)

svm = SVM(X, Y, 50.0, 1.0)
plane = svm['x'][: ccentre1.size]
print ('[SVM HPLANE]', plane)

point = intersection(ccentre1, ccentre2, plane)
print ('[IPOINT]', point)