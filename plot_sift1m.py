import matplotlib.pyplot as plt

q1_y = [0.43, 0.41, 0.39, 0.37, 0.36, 0.35]
q2_y = [0.5, 0.48, 0.46, 0.44, 0.43, 0.42]
q3_y = [0.58, 0.55, 0.54, 0.52, 0.51, 0.49]

x = ['Core', '+20%', '+40%', '+60%', '+80%', '+100%']

q1 = plt.scatter(x, q1_y, marker='x')
q2 = plt.scatter(x, q2_y, marker='o')
q3 = plt.scatter(x, q3_y, marker='d')

plt.axhline(y=0.43, color='r', linestyle='--')

plt.legend([q3, q2, q1], ['q3', 'q2', 'q1'])
plt.title('SIFT_1M T100 Precision')
plt.show()
