DISCRIMINATOR: Labels an image if generated or not
GENERATOR: Generates an image which tries to fool the discriminator

The generator's loss was not converging untill batch normalization and layer normalization was added to convolution and linear layers of the discriminator, respectively. Injecting gaussian noise to discriminator's input also helped.