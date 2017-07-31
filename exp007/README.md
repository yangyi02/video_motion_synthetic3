## Same as Exp004 But Use Pixel Average for Occlusion Place 

- No Occlusion modeling, predict conflict pixel values as the average between all coming pixels
- Predict motion for every pixel
- Photometric loss for pixels that are not occluded, then divided by the total number of existing pixels

### Results

- Column 2 and 3: Improve reconstruction loss over baseline (%) 

| Exp  | Test | Gt   | EPE  | Note |
| ---- | ---- | ---- | ---- | ---- | 
| 01 | 81 | 77 | 0.03 | box, m_range=1 |
| 02 | 80 | 85 | 0.03 | mnist, m_range=1 |
| 03 | 88 | 92 | 0.11 | box, m_range=1, bg_move |
| 04 | 86 | 89 | 0.12 | mnist, m_range=1, bg_move |
| 05 | 74 | 71 | 0.08 | box, m_range=1, num_objects=2 |
| 06 | 75 | 80 | 0.07 | mnist, m_range=1, num_objects=2 |
| 07 | 86 | 79 | 0.06 | box, m_range=2 |
| 08 | 84 | 84 | 0.06 | mnist, m_range=2 |
| 09 | 89 | 91 | 0.19 | box, m_range=2, bg_move |
| 10 | 87 | 88 | 0.23 | mnist, m_range=2, bg_move |
| 11 | 80 | 76 | 0.14 | box, m_range=2, num_objects=2 |
| 12 | 78 | 81 | 0.15 | mnist, m_range=2, num_objects=2 |
| 13 |  |  |  | box, m_range=2, image_size=64 |
| 14 |  |  |  | mnist, m_range=2, image_size=64 |
| 15 |  |  |  | box, m_range=2, image_size=64, bg_move |
| 16 |  |  |  | mnist, m_range=2, image_size=64, bg_move |
| 17 |  |  |  | box, m_range=2, num_objects=2, image_size=64 |
| 18 |  |  |  | mnist, m_range=2, num_objects=2, image_size=64 |
| 19 |  |  |  | box, m_range=2, num_objects=2, num_frame=4 |

### Take Home Message

- Slightly worse than Exp004
- We still need a better model handling occlusion confliction.
