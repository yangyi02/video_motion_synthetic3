### Endpoint Error Results Summary

| Exp  | 01   | 02   | 03   | 04   | 05   | 06   | 07   | 08   | 09   | 10   | 11   | 12   | Comp | Eval | Note |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 001  | 0.06 | 0.06 | 0.14 | 0.13 | 0.14 | 0.12 | 0.13 | 0.12 | 0.24 | 0.24 | 0.26 | 0.23 | ---- | ---- | The simplest baseline |
| 002  | 0.03 | 0.03 | 0.12 | 0.13 | 0.09 | 0.08 | 0.07 | 0.06 | 0.23 | 0.25 | 0.18 | 0.16 | 001  | Good | Moving pixel occlude static pixel |
| 003  | 0.04 | 0.03 | 0.11 | 0.12 | 0.08 | 0.07 | 0.29 | 0.25 | 0.32 | 0.37 | 0.11 | 0.23 | 002  | Bad  | New appear pixel not in loss |
| 004  | 0.03 | 0.03 | 0.11 | 0.11 | 0.08 | 0.06 | 0.04 | 0.04 | 0.18 | 0.23 | 0.10 | 0.11 | 002  | Good | Old pixel loss divided by total number of old pixels | 
| 005  | 0.04 | 0.03 | 0.14 | 0.14 | 0.10 | 0.12 | 0.06 | 0.06 | 0.24 | 0.22 | 0.16 | 0.22 | 004  | Bad  | New appear and occlude location both not in loss |
| 006  | 0.08 | 0.05 | 0.15 | 0.14 | 0.14 | 0.12 | 0.08 | 0.06 | 0.25 | 0.21 | 0.20 | 0.17 | 004  | Bad  | Neural net predict disappear |
| 007  | 0.03 | 0.03 | 0.11 | 0.12 | 0.08 | 0.07 | 0.06 | 0.06 | 0.19 | 0.23 | 0.14 | 0.15 | 004  | Bad  | Use avearge value at occlusion |

| 008  | 0.05 | 0.05 | 0.14 | 0.13 | 0.12 | 0.12 | 0.08 | 0.18 | 0.24 | 0.25 | 0.22 | 0.35 | 004  | Bad  | Predict relative depth |
| 009  |  |  |  |  |  |  |  |  |  |  |  |  |   |  | Predict relative depth using only one image |
| 010  |  |  |  |  |  |  |  |  |  |  |  |  |   |  | Add segmentation temporal consistency loss |

| 009  |  |  |  |  |  |  |  |  |  |  |  |  |   |  | Decompose x and y |
| 010  |  |  |  |  |  |  |  |  |  |  |  |  |   |  | Wider network, proves exp008 is bad |
| 011  |  |  |  |  |  |  |  |  |  |  |  |  |   |  | Wider network |
| 014  |  |  |  |  |  |  |  |  |  |  |  |  |   |  | Predict relative depth |
| 015  |  |  |  |  |  |  |  |  |  |  |  |  |   |  | Old pixel loss divided by total number of old pixels |
| 016  |  |  |  |  |  |  |  |  |  |  |  |  |   |  | Bidirectional model |
| 017  |  |  |  |  |  |  |  |  |  |  |  |  |   |  | Add a few more layers at the bottom of neural net |
| 018  |  |  |  |  |  |  |  |  |  |  |  |  |   |  | Predict depth using only one image |
| 019  |  |  |  |  |  |  |  |  |  |  |  |  |   |  | Add segmentation temporal consistency loss |
| 020  |  |  |  |  |  |  |  |  |  |  |  |  |   |  | Bidirectional model |
| 000  | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 000  |      | |

| Exp  | 01   | 02   | 03   | 04   | 05   | 06   | 07   | 08   | 09   | 10   | 11   | 12   | Comp | Eval | Note |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| |  |  |  |  |  |  |  |  |  |  |  |  | |  | The simplest baseline |
| |  |  |  |  |  |  |  |  |  |  |  |  | |  | Moving pixel occlude static pixel |
| |  |  |  |  |  |  |  |  |  |  |  |  | |  | Old pixel loss divided by total number of old pixels |
| |  |  |  |  |  |  |  |  |  |  |  |  | |  | Old pixel loss divided by total number of old pixels | 
| |  |  |  |  |  |  |  |  |  |  |  |  | |  | Wider network |
| |  |  |  |  |  |  |  |  |  |  |  |  | |  | Old pixel loss divided by total number of old pixels |
| |  |  |  |  |  |  |  |  |  |  |  |  | |  | Extra loss for total number of new and conflicting pixels |


### Take Home Message

- We should use: new appear pixel not in loss, loss divided by total number of old pixels
