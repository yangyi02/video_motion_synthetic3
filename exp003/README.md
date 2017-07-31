## Same as Exp002 But Loss Consider New Appear Pixels

- Occlusion modeling, moving pixels will occlude static pixels
- Predict motion for every pixel
- Photometric loss for pixels that are not occluded

### Results

- Column 2 and 3: Improve reconstruction loss over baseline (%) 

| Exp  | Test | Gt   | EPE  | Note |
| ---- | ---- | ---- | ---- | ---- |
| 01 | 83 | 79 | 0.04 | box, m_range=1 |
| 02 | 81 | 87 | 0.03 | mnist, m_range=1 |
| 03 | 88 | 92 | 0.11 | box, m_range=1, bg_move |
| 04 | 86 | 90 | 0.12 | mnist, m_range=1, bg_move |
| 05 | 76 | 73 | 0.08 | box, m_range=1, num_objects=2 |
| 06 | 75 | 81 | 0.07 | mnist, m_range=1, num_objects=2 |
| 07 | 89 | 90 | 0.29 | box, m_range=2 |
| 08 | 87 | 92 | 0.25 | mnist, m_range=2 |
| 09 | 89 | 91 | 0.32 | box, m_range=2, bg_move |
| 10 | 87 | 88 | 0.37 | mnist, m_range=2, bg_move |
| 11 | 78 | 83 | 0.11 | box, m_range=2, num_objects=2 |
| 12 | 80 | 86 | 0.23 | mnist, m_range=2, num_objects=2 |
| 13 |  |  |  | box, m_range=2, image_size=64 |
| 14 |  |  |  | mnist, m_range=2, image_size=64 |
| 15 |  |  |  | box, m_range=2, image_size=64, bg_move |
| 16 |  |  |  | mnist, m_range=2, image_size=64, bg_move |
| 17 |  |  |  | box, m_range=2, num_objects=2, image_size=64 |
| 18 |  |  |  | mnist, m_range=2, num_objects=2, image_size=64 |
| 19 |  |  |  | box, m_range=2, num_objects=2, num_frame=4 |

### Take Home Message

- Although this significantly improves test loss, the optical flow estimation actually becomes much worse when motion range is larger than 1.
- It is interesting to see when motion range is limited to 1, this helps on motion prediction.
- However, once motion range is increased to 2, the model significantly degenerate to a very bad local optimal.
