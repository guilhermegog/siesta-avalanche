from effnet import MobNet_ClassifierF, MobNet_ClassifierG

print("#### CLASSIFIER G ######")
model = MobNet_ClassifierG(latent_layer=8)
print('##### CLASSIFIER F #####')
model = MobNet_ClassifierF(latent_layer=8, num_classes=100)
