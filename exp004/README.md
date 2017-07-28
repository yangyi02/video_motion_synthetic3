## Same as Exp002 But Loss Divided by Total Number of Old Pixels

- Occlusion modeling, moving pixels will occlude static pixels
- Predict motion for every pixel
- Photometric loss for pixels that are not occluded, then divided by the total number of existing pixels

### Results

- Column 2 and 3: Improve reconstruction loss over baseline (%) 

| Exp  | Test | Gt   | EPE  | Note |
| ---- | ---- | ---- | ---- | ---- | 
| 01 | 0.86 | 80 | 0.03 | box, m_range=1 |
| 02 | 0.81 | 87 | 0.03 | mnist, m_range=1 |
| 03 | 0.88 | 92 | 0.11 | box, m_range=1, bg_move |
| 04 | 0.86 | 89 | 0.11 | mnist, m_range=1, bg_move |
| 05 | 0.75 | 73 | 0.08 | box, m_range=1, num_objects=2 |
| 06 | 0.74 | 81 | 0.06 | mnist, m_range=1, num_objects=2 |
| 07 | 0.89 | 89 | 0.04 | box, m_range=2 |
| 08 | 0.86 | 92 | 0.04 | mnist, m_range=2 |
| 09 | 0.88 | 90 | 0.18 | box, m_range=2, bg_move |
| 10 | 0.86 | 87 | 0.23 | mnist, m_range=2, bg_move |
| 11 | 0.78 | 82 | 0.10 | box, m_range=2, num_objects=2 |
| 12 | 0.77 | 85 | 0.11 | mnist, m_range=2, num_objects=2 |
| 13 |  |  |  | box, m_range=2, image_size=64 |
| 14 |  |  |  | mnist, m_range=2, image_size=64 |
| 15 |  |  |  | box, m_range=2, image_size=64, bg_move |
| 16 |  |  |  | mnist, m_range=2, image_size=64, bg_move |
| 17 |  |  |  | box, m_range=2, num_objects=2, image_size=64 |
| 18 |  |  |  | mnist, m_range=2, num_objects=2, image_size=64 |
| 19 |  |  |  | box, m_range=2, num_objects=2, num_frame=4 |

### Take Home Message

- So far the best loss I have found for 3 frames
