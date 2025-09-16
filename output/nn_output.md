# Estimation of neural networks

### Week 2

Our own full implementation (model, loss, backprop, update):

```bash
❯ python scripts/week2_network.py --batch-size 32 --lr 3.0 --epochs 8
Training net...
Epoch 0 : Train loss: 0.3787, Val loss: 0.2682, Val acc: 0.9263
Epoch 1 : Train loss: 0.2279, Val loss: 0.2400, Val acc: 0.9338
Epoch 2 : Train loss: 0.1904, Val loss: 0.2136, Val acc: 0.9412
Epoch 3 : Train loss: 0.1745, Val loss: 0.2108, Val acc: 0.9429
Epoch 4 : Train loss: 0.1533, Val loss: 0.3352, Val acc: 0.9000
Epoch 5 : Train loss: 0.1400, Val loss: 0.2200, Val acc: 0.9376
Epoch 6 : Train loss: 0.1332, Val loss: 0.2190, Val acc: 0.9417
Epoch 7 : Train loss: 0.1224, Val loss: 0.2091, Val acc: 0.9434
Final model: Test loss: 0.1959, Test acc: 0.9458
```

### Week 3

PyTorch with autodiff:

```bash
❯ python scripts/week3_torch.py --batch-size 32 --lr 3.0 --epochs 8
Using cpu device
Training net...
Epoch 0 : Train loss: 0.3748, Val loss: 0.2566, Val acc: 0.9284
Epoch 1 : Train loss: 0.2240, Val loss: 0.2146, Val acc: 0.9389
Epoch 2 : Train loss: 0.1881, Val loss: 0.2039, Val acc: 0.9448
Epoch 3 : Train loss: 0.1692, Val loss: 0.2197, Val acc: 0.9398
Epoch 4 : Train loss: 0.1521, Val loss: 0.1895, Val acc: 0.9478
Epoch 5 : Train loss: 0.1397, Val loss: 0.2624, Val acc: 0.9248
Epoch 6 : Train loss: 0.1297, Val loss: 0.2590, Val acc: 0.9310
Epoch 7 : Train loss: 0.1269, Val loss: 0.1998, Val acc: 0.9471
Final model: Test loss: 0.1964, Test acc: 0.9456
```

---

**Side quest**: find the reason/explanation for the discrepancy between the two.
