## Model Relative Occlusion in Each Pixel Neighborhood

- Occlusion modeling, predict relative depth to help figure out occlusion
- Predict motion for every pixel
- Photometric loss for pixels that are not occluded, then divided by the total number of existing pixels

### Results

- Column 2 and 3: Improve reconstruction loss over baseline (%) 

| Exp  | Test | Gt   | EPE  | Note |
| ---- | ---- | ---- | ---- | ---- | 
| 01 | 86 |  | 0.05 | box, m_range=1 |
| 02 | 81 |  | 0.05 | mnist, m_range=1 |
| 03 | 87 |  | 0.14 | box, m_range=1, bg_move |
| 04 | 85 |  | 0.13 | mnist, m_range=1, bg_move |
| 05 | 75 |  | 0.12 | box, m_range=1, num_objects=2 |
| 06 | 75 |  | 0.12 | mnist, m_range=1, num_objects=2 |
| 07 | 90 |  | 0.08 | box, m_range=2 |
| 08 | 87 |  | 0.18 | mnist, m_range=2 |
| 09 | 87 |  | 0.24 | box, m_range=2, bg_move |
| 10 | 85 |  | 0.25 | mnist, m_range=2, bg_move |
| 11 | 79 |  | 0.22 | box, m_range=2, num_objects=2 |
| 12 | 80 |  | 0.35 | mnist, m_range=2, num_objects=2 |
| 13 |    |  |      | box, m_range=2, image_size=64 |
| 14 |    |  |      | mnist, m_range=2, image_size=64 |
| 15 |    |  |      | box, m_range=2, image_size=64, bg_move |
| 16 |    |  |      | mnist, m_range=2, image_size=64, bg_move |
| 17 |    |  |      | box, m_range=2, num_objects=2, image_size=64 |
| 18 |    |  |      | mnist, m_range=2, num_objects=2, image_size=64 |
| 19 |    |  |      | box, m_range=2, num_objects=2, num_frame=4 |

### Take Home Message

