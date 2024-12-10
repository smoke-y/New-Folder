paper: https://arxiv.org/pdf/1704.04861

OVERVIEW:
Make convolution operation cheaper by factoring the convolution operation over multiple channels to convolution operation on each channel and using 1x1 convolution operation to combine all the results.

NOTE_TO_SELF:
maybe if i had used softmax in the end, my training would have been easier?
Cross entropy loss function compares 2 probability distribution, hence when 1(ground truth) is in between 0 and 1, keeping other also between 0 and 1 would be better.

TRAINED_MODEL: https://huggingface.co/0VISH0/Mobile-Net

Convolution Cost:

Kx * Ky * N * M * Ix * Iy

where, Kx = width of a kernel   <br>
       Ky = height of a kernel  <br>
       N  = number of kernels   <br>
       M  = channels of input   <br>
       Ix = width of input      <br>
       Iy = height of input

Depthwise seperable convolution cost:

Kx * Ky * Ix * Iy * M + M * N * Ix * Iy