'''
This code allow us to plot the ROC curve for our experiment

upd. 21.03.16 - Code re-structured for modularity + added ratios
upd 29.03.16 - added compcost to compare losses
'''
import cPickle
import matplotlib.pylab as plt
import numpy as np
import network_interson
f = open('net_SinEvid2_0.001_0.001_100_0.5.pkl','rb')
net = cPickle.load(f)
f.close()

true_positive = net.TP
false_positive = net.FP
false_negative = net.FN
true_negative = net.TN
mcc = net.mcc
total = len(true_positive) + len(true_negative) + len(false_positive) + len(false_negative)
sensitivity = []
specificity = []
for i in xrange(len(true_positive)):
     sensitivity.append(true_positive[i]/(true_positive[i] + false_negative[i]))
     specificity.append(true_negative[i]/(true_negative[i] + false_positive[i]))

for i in xrange(len(specificity)):
     specificity[i] = 1 - specificity[i]
foo = raw_input('What would you like to print? \n {cost // ROC // sens/specif ratio (ratio1) // TP,TN,FP,FN ratio (ratio2) // compare costs (compcost)} \n')

if foo == 'cost':
	print 'Plotting loss vs. iteration. If this is not desired, then modify plot_network.py'
	cost = net.cost_train
	plt.plot(cost,'bo')
	plt.xlabel('Iteration')
	plt.ylabel('Loss')
	plt.show()

if foo == 'compcost':
	print 'Comparing network losses vs. iteration. If this is not desired, then modify plot_network.py'
	
	cost = net.cost_train

	f = open('net_shuffle2_0.001_0.001_100_0.5.pkl','rb')
	cost2 = cPickle.load(f).cost_train
	f.close()

	#t = range(len(cost))
	#plt.plot(t, 'ko', label='len(cost)')	
	#t2 = range(len(cost2))
	#plt.plot(t2, 'ro', label='len(cost2)')

	plt.plot(cost, 'ro', label='880 vs 4000')
	plt.plot(cost2, 'bo',label='Shuffled')


	plt.xlabel('Iteration')
	plt.ylabel('Loss')
	plt.legend(loc='upper right')
	plt.show()


if foo == 'ROC':
	print 'Plotting the ROC. If this is not desired, then modify plot_network.py'

	plt.plot(specificity, sensitivity)
	plt.axis([0.0, 0.15, 0.0, 1.0])
	plt.xlabel('1-Specificity')
	plt.ylabel('Sensitivity')
	plt.show()

if foo == 'ratio1':
	print 'Plotting the sensitivity/specificity ratio. If this is not desired, then modify plot_network.py'
	ratio = []
	for i in xrange(len(sensitivity)):
		try:
			ratio.append(sensitivity[i]/specificity[i])
		except ZeroDivisionError:
			ratio.append(0)
	plt.plot(ratio)
	plt.xlabel('Iteration')
	plt.ylabel('sensitivity/specificity')
	plt.show()

if foo == 'ratio2':
	TPratio = []
	FPratio = []
	TNratio = []
	FNratio = []
	for i in xrange(len(true_positive)):
		try:
			TPratio.append(true_positive[i]/total)
			FPratio.append(false_positive[i]/total)
			TNratio.append(true_negative[i]/total)
			FNratio.append(true_positive[i]/total)
		except ZeroDivisionError:
			pass
	plt.plot(TPratio, label='TP ratio')
	plt.plot(FPratio, label='FP ratio')
	plt.plot(TNratio, label='TN ratio')
	plt.plot(FNratio, label='FN ratio')
	plt.xlabel('Epoch')
	plt.ylabel('ratio')
	plt.grid(True)
	plt.show()
		
	


