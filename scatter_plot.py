#scatter_plot
import matplotlib.pyplot as plt
import csv

math_score=[]
read_score=[]
with open('StudentsPerformance.csv','r') as csvfile:
    plots=csv.reader(csvfile,delimiter=',')
    for column in plots:
        math_score.append(str(column[5]))
        read_score.append(str(column[6]))

figure=plt.figure()
plt.axis([0,100,0,100])
plt.scatter(math_score,read_score,color='r')
##plt.plot(math_score,read_score,color="blue",markeredgecolor="black")
plt.title("Student score")
plt.xlabel("math score")
plt.ylabel("reading score")
plt.legend()
plt.show()
figure.savefig("scatterplot.svg")

