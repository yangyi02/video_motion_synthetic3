## Similar as Exp001 but Reconstruction Consider Moving Pixels Occlude Static Pixels

- Occlusion modeling, moving pixels will occlude static pixels
- Predict motion for every pixel
- Photometric loss for every pixel

### Results

- Column 2 and 3: Improve reconstruction loss over baseline (%) 

| Exp  | Test | Gt   | EPE  | Note |
| ---- | ---- | ---- | ---- | ---- | 
| 01 | 71 | 64 | 0.03 | box, m_range=1 |
| 02 | 74 | 77 | 0.03 | mnist, m_range=1 |
| 03 | 87 | 89 | 0.12 | box, m_range=1, bg_move |
| 04 | 84 | 86 | 0.13 | mnist, m_range=1, bg_move |
| 05 | 63 | 50 | 0.09 | box, m_range=1, num_objects=2 |
| 06 | 68 | 69 | 0.08 | mnist, m_range=1, num_objects=2 |
| 07 | 73 | 67 | 0.07 | box, m_range=2 |
| 08 | 77 | 78 | 0.06 | mnist, m_range=2 |
| 09 | 86 | 84 | 0.23 | box, m_range=2, bg_move |
| 10 | 84 | 81 | 0.25 | mnist, m_range=2, bg_move |
| 11 | 63 | 51 | 0.18 | box, m_range=2, num_objects=2 |
| 12 | 69 | 68 | 0.16 | mnist, m_range=2, num_objects=2 |
| 13 |  |  |  | box, m_range=2, image_size=64 |
| 14 |  |  |  | mnist, m_range=2, image_size=64 |
| 15 |  |  |  | box, m_range=2, image_size=64, bg_move |
| 16 |  |  |  | mnist, m_range=2, image_size=64, bg_move |
| 17 |  |  |  | box, m_range=2, num_objects=2, image_size=64 |
| 18 |  |  |  | mnist, m_range=2, num_objects=2, image_size=64 |
| 19 |  |  |  | box, m_range=2, num_objects=2, num_frame=4 |

### Take Home Message

- Although this is not a perfect model, but significantly than baseline exp001, even when background moves or 2 objects overlap.
- Visualization suggests this is a good baseline.
