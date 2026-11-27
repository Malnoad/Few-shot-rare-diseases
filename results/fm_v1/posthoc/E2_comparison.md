# E2 external error analysis

post hoc, Phase-1 rule E2 timestamped by commit 4b4e883 (20 Nov 2026), computed 2026-11-21; outside the locked protocol; never replaces a locked result

45 of 133 reported E2d intervals exclude 0.5.

## Locked-feature OCTDL normals

| Arm | Property | Failures | Common disease | False rare | AUROC | 95% interval |
|---|---|---|---|---|---|---|
| RETFound | width | 195 | 177 | 18 | 0.421 | 0.337 to 0.509 |
| RETFound | height | 195 | 177 | 18 | 0.459 | 0.388 to 0.538 |
| RETFound | aspect_ratio | 195 | 177 | 18 | 0.375 | 0.304 to 0.449 |
| RETFound | border_share | 195 | 177 | 18 | 0.500 | 0.500 to 0.500 |
| RETFound | near_white_share | 195 | 177 | 18 | 0.637 | 0.552 to 0.716 |
| RETFound | grey_mean | 195 | 177 | 18 | 0.807 | 0.749 to 0.859 |
| RETFound | grey_sd | 195 | 177 | 18 | 0.669 | 0.596 to 0.738 |
| DINOv2 | width | 187 | 46 | 141 | 0.501 | 0.421 to 0.585 |
| DINOv2 | height | 187 | 46 | 141 | 0.519 | 0.449 to 0.591 |
| DINOv2 | aspect_ratio | 187 | 46 | 141 | 0.478 | 0.406 to 0.549 |
| DINOv2 | border_share | 187 | 46 | 141 | 0.500 | 0.500 to 0.500 |
| DINOv2 | near_white_share | 187 | 46 | 141 | 0.509 | 0.431 to 0.591 |
| DINOv2 | grey_mean | 187 | 46 | 141 | 0.586 | 0.510 to 0.657 |
| DINOv2 | grey_sd | 187 | 46 | 141 | 0.575 | 0.505 to 0.642 |
| MIRAGE | width | 306 | 276 | 30 | 0.515 | 0.382 to 0.659 |
| MIRAGE | height | 306 | 276 | 30 | 0.449 | 0.327 to 0.581 |
| MIRAGE | aspect_ratio | 306 | 276 | 30 | 0.642 | 0.546 to 0.736 |
| MIRAGE | border_share | 306 | 276 | 30 | 0.500 | 0.500 to 0.500 |
| MIRAGE | near_white_share | 306 | 276 | 30 | 0.416 | 0.317 to 0.534 |
| MIRAGE | grey_mean | 306 | 276 | 30 | 0.635 | 0.510 to 0.752 |
| MIRAGE | grey_sd | 306 | 276 | 30 | 0.726 | 0.622 to 0.827 |
| Inception | width | 266 | 255 | 11 | 0.436 | 0.347 to 0.530 |
| Inception | height | 266 | 255 | 11 | 0.516 | 0.426 to 0.605 |
| Inception | aspect_ratio | 266 | 255 | 11 | 0.429 | 0.339 to 0.527 |
| Inception | border_share | 266 | 255 | 11 | 0.500 | 0.500 to 0.500 |
| Inception | near_white_share | 266 | 255 | 11 | 0.500 | 0.410 to 0.587 |
| Inception | grey_mean | 266 | 255 | 11 | 0.584 | 0.496 to 0.670 |
| Inception | grey_sd | 266 | 255 | 11 | 0.535 | 0.450 to 0.621 |

E2a records all destinations, E2b the overlap across arms, and E2c the locked-to-E1-M transitions.
The intervals describe associations among these scans. E2f uses one non-clinician rater.
