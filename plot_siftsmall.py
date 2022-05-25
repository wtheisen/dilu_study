import matplotlib.pyplot as plt

q1_y = [0.73, 0.69, 0.66, 0.65, 0.637, 0.62]
q2_y = [0.77, 0.75, 0.72, 0.7, 0.685, 0.68]
q3_y = [0.83, 0.82, 0.8, 0.77, 0.76, 0.75]

x = ['Core', '+20%', '+40%', '+60%', '+80%', '+100%']

q1 = plt.scatter(x, q1_y, marker='x')
q2 = plt.scatter(x, q2_y, marker='o')
q3 = plt.scatter(x, q3_y, marker='d')

plt.axhline(y=0.70, color='r', linestyle='--')

plt.legend([q3, q2, q1], ['q3', 'q2', 'q1'])
plt.title('SIFT_small T100 Precision')
plt.show()
