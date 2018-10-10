import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model

img_width, img_height = 150, 150
model_path = './newmodels/model.h5'
model_weights_path = './newmodels/weights.h5'
model = load_model(model_path)
model.load_weights(model_weights_path)

def predict(file):
  x = load_img(file, target_size=(img_width,img_height))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  array = model.predict(x)
  result = array[0]
  answer = np.argmax(result)
  if answer == 0:
    print("Pred Label: Aeolian")
    print("----")
  elif answer == 1:
    print("Pred Label : Dry")
    print("----")
  elif answer == 2:
    print("Pred Label: Glacial")
    print("----")
  elif answer == 3:
    print("Pred Label : Volcanic")
    print("----")
  return answer

aeolian_t = 0
aeolian_f = 0
dry_t = 0
dry_f = 0
glacial_t = 0
glacial_f = 0
volcanic_t = 0
volcanic_f = 0

for i, ret in enumerate(os.walk('./testdata_NZ/aeolian')):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue
    print("True Label: Aeolian")
    result = predict(ret[0] + '/' + filename)
    if result == 0:
      aeolian_t += 1
    else:
      aeolian_f += 1

for i, ret in enumerate(os.walk('./testdata_NZ/dry')):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue
    print("True Label: Dry")
    result = predict(ret[0] + '/' + filename)
    if result == 1:
      dry_t += 1
    else:
      dry_f += 1

for i, ret in enumerate(os.walk('./testdata_NZ/glacial')):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue
    print("True Label: Glacial")
    result = predict(ret[0] + '/' + filename)
    if result == 2:
      glacial_t += 1
    else:
      glacial_f += 1


for i, ret in enumerate(os.walk('./testdata_NZ/volcanic')):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue
    print("True Label: volcanic")
    result = predict(ret[0] + '/' + filename)
    if result == 3:
      volcanic_t += 1
    else:
      volcanic_f += 1

"""
Check metrics
"""
print("True Aeolian: ", aeolian_t)
print("False Aeolian: ", aeolian_f)
print("True Dry: ", dry_t)
print("False Dry: ", dry_f)
print("True Glacial: ", glacial_t)
print("False Glacial: ", glacial_f)
print("True Volcanic: ", volcanic_t)
print("False Volcanic: ", volcanic_f)