import numpy as np
import torch
from torch.nn import NLLLoss

from model.siesta_class import SiestaClassifier

n_features = 2

n_samples = 100
n_classes = 4

classifier = SiestaClassifier(n_features, n_classes)

x = []
angle = np.pi / 4
for i in range(n_classes):
    mu = np.ones((n_samples, n_features))
    if i != 0:
        rotate_vector = np.array([np.cos(angle), np.sin(angle)])
        mu = mu * rotate_vector
    angle += np.pi / 2
    x.append(np.random.normal(mu, 0.01, (n_samples, n_features)))
# Test the forward method

acc = 0
cnt1 = 0
cnt2 = 0
loss = NLLLoss()

for i in range(n_samples):
    true_y = torch.randint(0, n_classes, (1,))
    dist = x[true_y]
    input = torch.tensor(dist[i, :]).float()
    input = input.reshape(n_features, 1)

    out, prob = classifier(input)
    pred_y = torch.argmax(prob)
    it_loss = loss(out, true_y)

    if pred_y == true_y:
        acc += 1

    debug = True
    if debug:
        print("--------------------")
        # print(prob)
        print(
            "Predicted prototype, sample difference",
            torch.linalg.vector_norm(
                (classifier.weights[:, pred_y] - input), ord=2
            ).item(),
        )
        print(f"Loss at iteration {i}: {it_loss} ")
        print(f"Accuracy at iteration {i}: {acc/(i+1)}")
        print("--------------------")

    classifier.online_update(input, true_y)

print(classifier.weights)
print(classifier.class_counter)
