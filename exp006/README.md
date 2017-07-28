## Same as Exp004 But Predict Occlusion Using Neural Nets 

- Occlusion modeling, predict occlusion using neural nets instead of derived from motion
- Predict motion for every pixel
- Photometric loss for pixels that are not occluded, then divided by the total number of existing pixels

### Results

- Column 2 and 3: Improve reconstruction loss over baseline (%) 
- The gt model should report 100% reconstruction accuracy, however, the GtNet is not since it does not really use depth

| Exp  | Test | Gt   | EPE  | Note |
| ---- | ---- | ---- | ---- | ---- | 
| 01 | 0.80 | 79 | 0.08 | box, m_range=1 |
| 02 | 0.78 | 87 | 0.05 | mnist, m_range=1 |
| 03 | 0.86 | 92 | 0.15 | box, m_range=1, bg_move |
| 04 | 0.84 | 89 | 0.14 | mnist, m_range=1, bg_move |
| 05 | 0.72 | 71 | 0.14 | box, m_range=1, num_objects=2 |
| 06 | 0.73 | 81 | 0.12 | mnist, m_range=1, num_objects=2 |
| 07 | 0.87 | 89 | 0.08 | box, m_range=2 |
| 08 | 0.85 | 92 | 0.06 | mnist, m_range=2 |
| 09 | 0.87 | 90 | 0.25 | box, m_range=2, bg_move |
| 10 | 0.86 | 86 | 0.21 | mnist, m_range=2, bg_move |
| 11 | 0.76 | 82 | 0.20 | box, m_range=2, num_objects=2 |
| 12 | 0.76 | 85 | 0.17 | mnist, m_range=2, num_objects=2 |
| 13 |  |  |  | box, m_range=2, image_size=64 |
| 14 |  |  |  | mnist, m_range=2, image_size=64 |
| 15 |  |  |  | box, m_range=2, image_size=64, bg_move |
| 16 |  |  |  | mnist, m_range=2, image_size=64, bg_move |
| 17 |  |  |  | box, m_range=2, num_objects=2, image_size=64 |
| 18 |  |  |  | mnist, m_range=2, num_objects=2, image_size=64 |
| 19 |  |  |  | box, m_range=2, num_objects=2, num_frame=4 |

### Take Home Message

- Using neural net to predict which will disappear in the next frame is not easy.
- We still need a better model on occlusion.
