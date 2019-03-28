transfer learning:
if you look at "deconvolutional neural nets", they show what filters are being learned for every layer in the CNN.
for first 10 epoch the first layer learn filters and then stop, for second 10 epoch second layer is learning filters and so on..
Transfer learning is the idea of, freezing the first n-layers that learn less-abstract filter, and then just train other "more abstract" layers on the top of those freezed layers.

image reconstruction:
[latent representation] deep layer retain lest content information and keep more abstract concepts
(learn an image and then start from random noise and see how much informatino the latent representation has retained)

texture generation:
given a trained network, and an image (with texture) A,  I can give random noise and use a "special" loss to update my random noise to resemble more and more my A


