paper: https://arxiv.org/pdf/1704.04861

OVERVIEW:
Make convolution operation cheaper by factoring the convolution operation over multiple channels to convolution operation on each channel and using 1x1 convolution operation to combine all the results.

NOTE_TO_SELF:
maybe if i had used softmax in the end, my accuracy would have increased?
Cross entropy loss function compares 2 probability distribution, hence when 1(ground truth) is in between 0 and 1, keeping other also between 0 and 1 would be better.