## The Simplest Version

- No occlusion modeling
- Predict motion for every pixel
- Photometric loss for every pixel

### Results

- Column 2 and 3: Improve reconstruction loss over baseline (%) 

| Exp  | Test | Gt   | EPE  | Note |
| ---- | ---- | ---- | ---- | ---- | 
| 01 |  | 61 | 0.07 | box, m_range=1 |
| 02 |  | 71 | 0.07 | mnist, m_range=1 |
| 03 |  | 85 | 0.08 | box, m_range=1, bg_move |
| 04 |  | 84 | 0.09 | mnist, m_range=1, bg_move |
| 05 |  | 43 | 0.15 | box, m_range=1, num_objects=2 |
| 06 |  | 61 | 0.14 | mnist, m_range=1, num_objects=2 |
| 07 |  | 57 | 0.11 | box, m_range=2 |
| 08 |  | 68 | 0.13 | mnist, m_range=2 |
| 09 |  | 78 | 0.18 | box, m_range=2, bg_move |
| 10 |  | 77 | 0.22 | mnist, m_range=2, bg_move |
| 11 |  | 36 | 0.26 | box, m_range=2, num_objects=2 |
| 12 |  | 58 | 0.26 | mnist, m_range=2, num_objects=2 |
| 13 |  | 63 | 0.11 | box, m_range=2, image_size=64 |
| 14 |  | 68 | 0.03 | mnist, m_range=2, image_size=64 |
| 15 |  | 87 |      | box, m_range=2, image_size=64, bg_move |
| 16 |  | 91 |      | mnist, m_range=2, image_size=64, bg_move |
| 17 |  | 44 |      | box, m_range=2, num_objects=2, image_size=64 |
| 18 |  | 66 |      | mnist, m_range=2, num_objects=2, image_size=64 |
| 19 |  | 40 |      | box, m_range=2, num_objects=2, num_frame=4 |

### Take Home Message

- Predicted motion improves reconstruction loss better than using ground truth motion, suggesting this is not a correct model.
- Visualization suggests the error occurs mostly at the occlusion boundary.
- Visualization suggests this is still a good baseline.
- The reason that EPE in exp14 is small is because the Mnist digits are very small compared to static background. 
