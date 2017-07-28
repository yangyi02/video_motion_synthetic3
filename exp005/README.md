## Same as Exp004 But Conflict Region and Appear Region Both no Loss

- Occlusion modeling, predict conflict occlusion and appear using motion
- Predict motion for every pixel
- Photometric loss for pixels that are not occluded and not appeared, then divided by the total number of existing pixels

### Results

- Column 2 and 3: Improve reconstruction loss over baseline (%) 

| Exp  | Test | Gt   | EPE  | Note |
| ---- | ---- | ---- | ---- | ---- | 
| 01 | 80 | 75 | 0.04 | box, m_range=1 |
| 02 | 78 | 86 | 0.03 | mnist, m_range=1 |
| 03 | 86 | 93 | 0.14 | box, m_range=1, bg_move |
| 04 | 84 | 91 | 0.14 | mnist, m_range=1, bg_move |
| 05 | 69 | 69 | 0.10 | box, m_range=1, num_objects=2 |
| 06 | 73 | 81 | 0.12 | mnist, m_range=1, num_objects=2 |
| 07 | 89 | 87 | 0.06 | box, m_range=2 |
| 08 | 88 | 92 | 0.06 | mnist, m_range=2 |
| 09 | 88 | 94 | 0.24 | box, m_range=2, bg_move |
| 10 | 86 | 93 | 0.22 | mnist, m_range=2, bg_move |
| 11 | 77 | 83 | 0.16 | box, m_range=2, num_objects=2 |
| 12 | 81 | 88 | 0.22 | mnist, m_range=2, num_objects=2 |
| 13 |  |  |  | box, m_range=2, image_size=64 |
| 14 |  |  |  | mnist, m_range=2, image_size=64 |
| 15 |  |  |  | box, m_range=2, image_size=64, bg_move |
| 16 |  |  |  | mnist, m_range=2, image_size=64, bg_move |
| 17 |  |  |  | box, m_range=2, num_objects=2, image_size=64 |
| 18 |  |  |  | mnist, m_range=2, num_objects=2, image_size=64 |
| 19 |  |  |  | box, m_range=2, num_objects=2, num_frame=4 |

### Take Home Message

- Not as good as Exp004 
- Perhaps we still need to reconsider bidirectional model or even deal with occlusion now.
