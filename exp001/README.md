## The Simplest Version

- No occlusion modeling
- Predict motion for every pixel
- Photometric loss for every pixel

### Results

- Column 2 and 3: Improve reconstruction loss over baseline (%) 

| Exp  | Test | Gt   | EPE  | Note |
| ---- | ---- | ---- | ---- | ---- | 
| 01 | 62 | 62 | 0.06 | box, m_range=1 |
| 02 | 70 | 71 | 0.06 | mnist, m_range=1 |
| 03 | 85 | 91 | 0.14 | box, m_range=1, bg_move |
| 04 | 83 | 87 | 0.14 | mnist, m_range=1, bg_move |
| 05 | 58 | 58 | 0.13 | box, m_range=1, num_objects=2 |
| 06 | 64 | 68 | 0.12 | mnist, m_range=1, num_objects=2 |
| 07 | 72 | 56 | 0.13 | box, m_range=2 |
| 08 | 75 | 65 | 0.12 | mnist, m_range=2 |
| 09 | 87 | 86 | 0.23 | box, m_range=2, bg_move |
| 10 | 84 | 81 | 0.24 | mnist, m_range=2, bg_move |
| 11 | 63 | 53 | 0.26 | box, m_range=2, num_objects=2 |
| 12 | 68 | 60 | 0.24 | mnist, m_range=2, num_objects=2 |
| 13 |  |  |  | box, m_range=2, image_size=64 |
| 14 |  |  |  | mnist, m_range=2, image_size=64 |
| 15 |  |  |  | box, m_range=2, image_size=64, bg_move |
| 16 |  |  |  | mnist, m_range=2, image_size=64, bg_move |
| 17 |  |  |  | box, m_range=2, num_objects=2, image_size=64 |
| 18 |  |  |  | mnist, m_range=2, num_objects=2, image_size=64 |
| 19 |  |  |  | box, m_range=2, num_objects=2, num_frame=4 |

### Take Home Message

- Predicted motion improves reconstruction loss better than using ground truth motion, suggesting this is not a correct model.
- Visualization suggests the error occurs mostly at the occlusion boundary.
- Visualization suggests this is still a good baseline.
- The reason that EPE in exp14 is small is because the Mnist digits are very small compared to static background. 
