import matplotlib.pyplot as plt
from network import restore_model
from network import generator

model, test_it = restore_model()

for input in generator(test_it):
    embedder_output, detector_output = model(input, training=False)
    break

plt.rcParams["figure.figsize"] = (5,2)
plt.subplot(2, 5, 1)
plt.imshow(embedder_output[0])
plt.axis('off')
plt.subplot(2, 5, 2)
plt.imshow(embedder_output[1])
plt.axis('off')
plt.subplot(2, 5, 3)
plt.imshow(embedder_output[2])
plt.axis('off')
plt.subplot(2, 5, 4)
plt.imshow(embedder_output[3])
plt.axis('off')
plt.subplot(2, 5, 5)
plt.imshow(embedder_output[4])
plt.axis('off')


plt.subplot(2, 5, 6)
plt.imshow(input[0][0])
plt.axis('off')
plt.subplot(2, 5, 7)
plt.imshow(input[0][1])
plt.axis('off')
plt.subplot(2, 5, 8)
plt.imshow(input[0][2])
plt.axis('off')
plt.subplot(2, 5, 9)
plt.imshow(input[0][3])
plt.axis('off')
plt.subplot(2, 5, 10)
plt.imshow(input[0][4])
plt.axis('off')
plt.savefig('images1.pdf')