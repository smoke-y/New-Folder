A neural net is nothing but a huge non-linear function. This huge non-linear function consists of small transformations from one dimension to another in a non-linear way.

```python
self.l1 = nn.Linear(2, 5)
self.l2 = nn.Linear(5, 1)
```

The first line converts the input to higher dimension and the second line converts it back to a low dimension tensor.

```python
x = torch.sigmoid(self.l1(x))
return torch.sigmoid(self.l2(x))
```

After each conversion, we apply a non-linear function. If we don't do this, the entire model would just be one big linear transformation. Which might be good enough when you are modeling functions such as OR, AND, etc...These gate's output can be seperated by 1 line(any input point above this line is 0 and anything below is 1 or vice verca) But XOR is too complicated to learned(seperated) by a straight(linear transformation) line.

Hence we do some non-linear transformations to a higher dimension hoping we can seperate the points in that higher dimension. Then we translate the answer back to lower dimension.

<img href="xor.png">