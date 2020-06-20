import numpy as np #numpy is used for a parameter input
from steganogan import SteganoGAN
from steganogan.loader import DataLoader
from steganogan.encoders import BasicEncoder, DenseEncoder
from steganogan.decoders import BasicDecoder, DenseDecoder
from steganogan.critics import BasicCritic

# Load the data
train = DataLoader('data/div2k/train/', limit=np.inf, shuffle=True, batch_size=4)
validation = DataLoader('data/div2k/val/', limit=np.inf, shuffle=True, batch_size=4)

# Create the SteganoGAN instance
steganogan = SteganoGAN(1, BasicEncoder, BasicDecoder, BasicCritic, hidden_size=32, cuda=True, verbose=True)

# Fit on the given data
steganogan.fit(train, validation, epochs=100)
# Save the fitted model
steganogan.save('model/demo.steg')

# Load the model
steganogan = SteganoGAN.load(architecture=None, path='model/demo.steg', cuda=True, verbose=True)
# Encode a message in input.png
steganogan.encode('data/mo.jpg', 'data/output.png', 'This is a super secret message!')
# Decode the message from output.png
steganogan.decode('data/output.png')