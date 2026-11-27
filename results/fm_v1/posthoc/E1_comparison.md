# E1 size and framing control (post hoc, Phase-1)

Label of every row: post hoc, Phase-1 rule E1 timestamped by commit 4b4e883 (20 Nov 2026), computed 2026-11-21; outside the locked protocol; never replaces a locked result; limitations: the centre square drops lateral retina unequally by source and class; MIRAGE now gets 256 px of detail enlarged to 512; E1-M does not remove effective resolution (content short side 103-496 px), field of view, scanner appearance, compression, intensity, black-clipped background against speckle, or Kermany's white fill wedges.

Only E1-M gives the verdict on M1; E1-L, E1-C, the references and the hold-out are descriptions and cannot overturn or rescue it. The development spread is over support draws on 124 fixed queries, not a confidence interval.

Pixel-only reference (choice fixed in code before any E number). "Is above" means p05 > 0 across the 100 paired MIRAGE E1-M minus E1-M thumbnail K-mean differences. The quoted mean is the mean of those differences.

## 1. Success table (E1d)

| Line | Set | Status | n | Mean | SD | p05 | Draws above 0 | Verdict |
|---|---|---|---|---|---|---|---|---|
| MIRAGE minus Inception-v3, K mean | locked | interpreted | 100 | 0.116 | 0.037 | 0.069 | 100 | description only |
| MIRAGE minus Inception-v3, K mean | E1-M | interpreted | 100 | 0.063 | 0.036 | 0.003 | 95 | pass |
| MIRAGE minus Inception-v3, K mean | E1-L | interpreted | 100 | 0.077 | 0.034 | 0.023 | 99 | description only |
| MIRAGE minus Inception-v3, K mean | E1-C | interpreted | 100 | 0.068 | 0.033 | 0.017 | 98 | description only |
| MIRAGE E1-M minus content_size reference (raw content box), K mean | E1-M | interpreted | 100 | 0.203 |  | 0.149 | 100 | companion line |
| MIRAGE E1-M minus thumbnail reference (E1-M), K mean | E1-M | interpreted | 100 | 0.290 |  | 0.243 | 100 | companion line |

**Sentence.** Under the size and framing control, MIRAGE keeps a positive development advantage over Inception-v3 (mean 0.063, locked 0.116; p05 0.003), and also when effective resolution is evened out (E1-L p05 = 0.023), and also when the image is only cropped to its content box (E1-C p05 = 0.017); it is above the pixel-only reference (mean 0.290).

**Limitations.** the centre square drops lateral retina unequally by source and class; MIRAGE now gets 256 px of detail enlarged to 512; E1-M does not remove effective resolution (content short side 103-496 px), field of view, scanner appearance, compression, intensity, black-clipped background against speckle, or Kermany's white fill wedges.

## 2. Checks (E1a)

| Check | Arm | Set | Quantity | Value | Status | Note |
|---|---|---|---|---|---|---|
| rule 1: Phase-1 subsection SHA256 |  |  | at rule commit 4b4e883 | 8b4bb0b22b0bf23c4cc488883248f3d8abc62afd8456fd4d8dcd6309ed7de958 | same | working 8b4bb0b22b0bf23c4cc488883248f3d8abc62afd8456fd4d8dcd6309ed7de958 |
| archive receipt |  |  | phase1-E1-C-DINOv2.zip | 49ed95aaa863609f0534e375da9b7acd2958664caa350781cdf4b2f1d8dbb196 | equal |  |
| archive receipt |  |  | phase1-E1-C-Inception.zip | ca5ab96d2bc6f4aa66866a4a32a075b8282e1c5f9b96b2373c108ff0aae4aa88 | equal |  |
| archive receipt |  |  | phase1-E1-C-MIRAGE.zip | 78fdeba1699832d2e047f1c3d3812f7e92b9cc628cd52ba043c43eacf34b977e | equal |  |
| archive receipt |  |  | phase1-E1-C-RETFound.zip | 0a799bb356843992db1d0d3bd456ad64887487e9847f3eb488ba84350ac56b75 | equal |  |
| archive receipt |  |  | phase1-E1-L-DINOv2.zip | 984f561da2bd49a85dde441e4b820902328e88ea7c487927c18e3d2b26222d68 | equal |  |
| archive receipt |  |  | phase1-E1-L-Inception.zip | 1d4872d6f4df402ba29ade01918888d1a58bdc9518665a2fb20433ac34d78ee3 | equal |  |
| archive receipt |  |  | phase1-E1-L-MIRAGE.zip | d1bcbe96a9ca9b40429aa65503bbf617a1e58f4eb468e69a3f0c20dfb60dde3c | equal |  |
| archive receipt |  |  | phase1-E1-L-RETFound.zip | 63bf4fff749a8355e1b5847da47526ea46f72bc5be9493d4a3ae8b4a0bacafea | equal |  |
| archive receipt |  |  | phase1-E1-M-DINOv2.zip | ee6986ba29618ed72dfa8648807413f9792bd46edf50947eba9bf8caf0461891 | equal |  |
| archive receipt |  |  | phase1-E1-M-Inception.zip | ab5d23be8b4478f20f1d3d81d7f3fdeea9e71fcfd8345d4c9bbc1679ee0cbd0e | equal |  |
| archive receipt |  |  | phase1-E1-M-MIRAGE.zip | fe645ceea8ed0a0e0fb2ef57dc4d199c6275eaa899004f8541cbab81f27383eb | equal |  |
| archive receipt |  |  | phase1-E1-M-RETFound.zip | 7782c610a9f6589b759ef29c47ec757374a9bf5f92de9107c8579fd973aff02f | equal |  |
| archive receipt |  |  | phase1-E3-MIRAGE-E3-ref.zip | 32540b6febf2c00958baaab475a66154cd897f521f7c2de7836a0c2b1062ad8f | equal |  |
| archive receipt |  |  | phase1-E3-MIRAGE-M-bicubic.zip | 413f5731fcd03bdee2d7e65aa72a8515d1db4f5077209669b209c23e97dbf4e3 | equal |  |
| archive receipt |  |  | phase1-E3-MIRAGE-M-crop.zip | 75db7f8db501fcca4cc7fe31902adf89e17ef534eb0bb7a023ea0bbe6c830650 | equal |  |
| archive receipt |  |  | phase1-E3-MIRAGE-M-nostretch.zip | a7da4833b4be56850fc1fc06fac84f7032ed6927c427ca8ffdb5d90792eb3a8a | equal |  |
| archive receipt |  |  | phase1-E3-MIRAGE-M-res256.zip | 8b4612ffcc886f18693590681614d2a55d896d47028b9468dd11bd4ba9a54128 | equal |  |
| archive receipt |  |  | phase1-E3-RETFound-E3-ref.zip | 630dba012456e1294c611a704f9f45dd5f9bb54da57501a7e0eaa6e9de5a5dcd | equal |  |
| archive receipt |  |  | phase1-E3-RETFound-R-bilinear.zip | 378c47733e328b072fbf45f2f7542c37b9a72a3329963b55114600c2a1bee90f | equal |  |
| archive receipt |  |  | phase1-E3-RETFound-R-norm05.zip | e532ab091719fcb15a4217fe5c01c552c9b4e2a90c8121b1d641fc493f047c6c | equal |  |
| archive receipt |  |  | phase1-E3-RETFound-R-pad.zip | 85cd9607683e5bb2f99b79e67ff83d69424e0151ce45fab05d00544168bccf77 | equal |  |
| archive receipt |  |  | phase1-E3-RETFound-R-res128.zip | a92e3b2cf15bfd0b2323a689f44d7df27e3abfe10384f346a89ae757f33aab96 | equal |  |
| archive receipt |  |  | phase1-E3-RETFound-R-squash.zip | 18ef6635441940b56d1babd38127ec1672cf8ab93d5247a50777b92b47b94356 | equal |  |
| archive receipt |  |  | phase1-E3-RETFound-R-stretch.zip | c67a23dc6042531754898c61c40e06ae6603da0376435e482463316385cdc922 | equal |  |
| archive receipt |  |  | phase1-render.zip | f28e8cb0f49402b72e4fbb4cfe4c528576e74626bbdc6d3f76f7123430046cf7 | equal |  |
| control: locked path against locked features | DINOv2 | E1-C | max_difference_ratio | 2.7778505118346576e-06 | pass | 5927 images; min cosine 1.000000 |
| repeat (recorded, not a gate) | DINOv2 | E1-C | max_difference_ratio | 0.0 |  | 5 images |
| identity | DINOv2 | E1-C | code_matches_lock | True |  |  |
| identity | DINOv2 | E1-C | environment_matches_lock | False |  |  |
| control: locked path against locked features | Inception | E1-C | max_difference_ratio | 1.4794658462535394e-06 | pass | 20 images; min cosine 1.000000 |
| repeat (recorded, not a gate) | Inception | E1-C | max_difference_ratio | 0.0 |  | 5 images |
| identity | Inception | E1-C | code_matches_lock | True |  |  |
| identity | Inception | E1-C | environment_matches_lock | True |  |  |
| control: locked path against locked features | MIRAGE | E1-C | max_difference_ratio | 1.3797153150607517e-07 | pass | 20 images; min cosine 1.000000 |
| repeat (recorded, not a gate) | MIRAGE | E1-C | max_difference_ratio | 0.0 |  | 5 images |
| identity | MIRAGE | E1-C | code_matches_lock | True |  |  |
| identity | MIRAGE | E1-C | environment_matches_lock | False |  |  |
| control: locked path against locked features | RETFound | E1-C | max_difference_ratio | 6.451012334787154e-08 | pass | 20 images; min cosine 1.000000 |
| repeat (recorded, not a gate) | RETFound | E1-C | max_difference_ratio | 0.0 |  | 5 images |
| identity | RETFound | E1-C | code_matches_lock | True |  |  |
| identity | RETFound | E1-C | environment_matches_lock | False |  |  |
| control: locked path against locked features | DINOv2 | E1-L | max_difference_ratio | 2.7778505118346576e-06 | pass | 5927 images; min cosine 1.000000 |
| repeat (recorded, not a gate) | DINOv2 | E1-L | max_difference_ratio | 0.0 |  | 5 images |
| identity | DINOv2 | E1-L | code_matches_lock | True |  |  |
| identity | DINOv2 | E1-L | environment_matches_lock | False |  |  |
| control: locked path against locked features | Inception | E1-L | max_difference_ratio | 1.6643990770352317e-06 | pass | 20 images; min cosine 1.000000 |
| repeat (recorded, not a gate) | Inception | E1-L | max_difference_ratio | 0.0 |  | 5 images |
| identity | Inception | E1-L | code_matches_lock | True |  |  |
| identity | Inception | E1-L | environment_matches_lock | True |  |  |
| control: locked path against locked features | MIRAGE | E1-L | max_difference_ratio | 1.3797153150607517e-07 | pass | 20 images; min cosine 1.000000 |
| repeat (recorded, not a gate) | MIRAGE | E1-L | max_difference_ratio | 0.0 |  | 5 images |
| identity | MIRAGE | E1-L | code_matches_lock | True |  |  |
| identity | MIRAGE | E1-L | environment_matches_lock | False |  |  |
| control: locked path against locked features | RETFound | E1-L | max_difference_ratio | 6.451012334787154e-08 | pass | 20 images; min cosine 1.000000 |
| repeat (recorded, not a gate) | RETFound | E1-L | max_difference_ratio | 0.0 |  | 5 images |
| identity | RETFound | E1-L | code_matches_lock | True |  |  |
| identity | RETFound | E1-L | environment_matches_lock | False |  |  |
| control: locked path against locked features | DINOv2 | E1-M | max_difference_ratio | 2.7778505118346576e-06 | pass | 5927 images; min cosine 1.000000 |
| repeat (recorded, not a gate) | DINOv2 | E1-M | max_difference_ratio | 0.0 |  | 5 images |
| identity | DINOv2 | E1-M | code_matches_lock | True |  |  |
| identity | DINOv2 | E1-M | environment_matches_lock | False |  |  |
| control: locked path against locked features | Inception | E1-M | max_difference_ratio | 1.0479549744295904e-06 | pass | 20 images; min cosine 1.000000 |
| repeat (recorded, not a gate) | Inception | E1-M | max_difference_ratio | 0.0 |  | 5 images |
| identity | Inception | E1-M | code_matches_lock | True |  |  |
| identity | Inception | E1-M | environment_matches_lock | True |  |  |
| control: locked path against locked features | MIRAGE | E1-M | max_difference_ratio | 1.3797153150607517e-07 | pass | 20 images; min cosine 1.000000 |
| repeat (recorded, not a gate) | MIRAGE | E1-M | max_difference_ratio | 0.0 |  | 5 images |
| identity | MIRAGE | E1-M | code_matches_lock | True |  |  |
| identity | MIRAGE | E1-M | environment_matches_lock | False |  |  |
| control: locked path against locked features | RETFound | E1-M | max_difference_ratio | 6.451012334787154e-08 | pass | 20 images; min cosine 1.000000 |
| repeat (recorded, not a gate) | RETFound | E1-M | max_difference_ratio | 0.0 |  | 5 images |
| identity | RETFound | E1-M | code_matches_lock | True |  |  |
| identity | RETFound | E1-M | environment_matches_lock | False |  |  |
| E3-ref full control | MIRAGE |  | max_difference_ratio | 2.992839213184965e-07 | pass | 5927 images |
| E3-ref full control | RETFound |  | max_difference_ratio | 1.9098830591335447e-07 | pass | 5927 images |
| Inception-v3 decoder change (PIL-decoded PNG originals) | Inception | E1-M | Kermany | 0.024556883026834245 |  | 20 images; min cosine 0.999642 |
| Inception-v3 decoder change (PIL-decoded PNG originals) | Inception | E1-M | OCTID | 0.020016204916140614 |  | 10 images; min cosine 0.999682 |
| Inception-v3 decoder change (PIL-decoded PNG originals) | Inception | E1-M | OCTDL | 0.04645150453702924 |  | 10 images; min cosine 0.997780 |
| Inception-v3 resize identity | Inception | E1-M | tf.image.resize | {"arrays": 6, "failures": 0, "unchanged": true} |  |  |
| render |  |  | images rendered (twice; saved PNG pixel hashes verified) | 5927 |  |  |
| render |  |  | content box fallback to the full frame (flagged) | 0 |  |  |
| render |  |  | files with alpha below 255 (dropped by convert('RGB')) | 0 |  |  |
| render |  |  | mode L, format JPEG | 5000 |  |  |
| render |  |  | mode RGB, format JPEG | 708 |  |  |
| render |  |  | mode RGB, format PNG | 11 |  |  |
| render |  |  | mode RGBA, format PNG | 208 |  |  |
| render | Kermany |  | content short-side range (px) | 240–512 |  | 5000 images |
| render | OCTDL |  | content short-side range (px) | 228–536 |  | 371 images |
| render | OCTID |  | content short-side range (px) | 488–586 |  | 410 images |
| render | Yoo rare |  | content short-side range (px) | 103–471 |  | 146 images |
| render | CSC | E1-L | content short side below 128 px | 2 |  |  |
| render | MH | E1-L | content short side below 128 px | 4 |  |  |
| render | Stargardt | E1-L | content short side below 128 px | 1 |  |  |
| framing residual | CSC | original | mean A2 border share | 0.45165301693899934 |  | n=32 |
| framing residual | CSC | original | share with A2 border share > 0 | 1.0 |  | n=32 |
| framing residual | CSC | original | share with near-black edge run > 5% of side | 1.0 |  | n=32 |
| framing residual | CSC | original | share with near-black edge run > 10% of side | 0.96875 |  | n=32 |
| framing residual | CSC | original | mean near-white share | 0.0004928912722263466 |  | n=32 |
| framing residual | CSC | E1-M | mean A2 border share | 0.04443359375 |  | n=32 |
| framing residual | CSC | E1-M | share with A2 border share > 0 | 0.5 |  | n=32 |
| framing residual | CSC | E1-M | share with near-black edge run > 5% of side | 0.46875 |  | n=32 |
| framing residual | CSC | E1-M | share with near-black edge run > 10% of side | 0.3125 |  | n=32 |
| framing residual | CSC | E1-M | mean near-white share | 0.00013875961303710938 |  | n=32 |
| framing residual | CSC | E1-L | mean A2 border share | 0.05224609375 |  | n=32 |
| framing residual | CSC | E1-L | share with A2 border share > 0 | 0.59375 |  | n=32 |
| framing residual | CSC | E1-L | share with near-black edge run > 5% of side | 0.5 |  | n=32 |
| framing residual | CSC | E1-L | share with near-black edge run > 10% of side | 0.375 |  | n=32 |
| framing residual | CSC | E1-L | mean near-white share | 7.200241088867188e-05 |  | n=32 |
| framing residual | CSC | E1-C | mean A2 border share | 0.0 |  | n=32 |
| framing residual | CSC | E1-C | share with A2 border share > 0 | 0.0 |  | n=32 |
| framing residual | CSC | E1-C | share with near-black edge run > 5% of side | 0.375 |  | n=32 |
| framing residual | CSC | E1-C | share with near-black edge run > 10% of side | 0.125 |  | n=32 |
| framing residual | CSC | E1-C | mean near-white share | 0.0006630585950266417 |  | n=32 |
| framing residual | Kermany | original | mean A2 border share | 0.02307676826395014 |  | n=5000 |
| framing residual | Kermany | original | share with A2 border share > 0 | 0.4648 |  | n=5000 |
| framing residual | Kermany | original | share with near-black edge run > 5% of side | 0.0054 |  | n=5000 |
| framing residual | Kermany | original | share with near-black edge run > 10% of side | 0.0016 |  | n=5000 |
| framing residual | Kermany | original | mean near-white share | 0.04154388437578755 |  | n=5000 |
| framing residual | Kermany | E1-M | mean A2 border share | 0.0033320281982421873 |  | n=5000 |
| framing residual | Kermany | E1-M | share with A2 border share > 0 | 0.3056 |  | n=5000 |
| framing residual | Kermany | E1-M | share with near-black edge run > 5% of side | 0.0066 |  | n=5000 |
| framing residual | Kermany | E1-M | share with near-black edge run > 10% of side | 0.0034 |  | n=5000 |
| framing residual | Kermany | E1-M | mean near-white share | 0.017842999267578125 |  | n=5000 |
| framing residual | Kermany | E1-L | mean A2 border share | 0.0054921875 |  | n=5000 |
| framing residual | Kermany | E1-L | share with A2 border share > 0 | 0.313 |  | n=5000 |
| framing residual | Kermany | E1-L | share with near-black edge run > 5% of side | 0.025 |  | n=5000 |
| framing residual | Kermany | E1-L | share with near-black edge run > 10% of side | 0.0108 |  | n=5000 |
| framing residual | Kermany | E1-L | mean near-white share | 0.015294351196289063 |  | n=5000 |
| framing residual | Kermany | E1-C | mean A2 border share | 0.0 |  | n=5000 |
| framing residual | Kermany | E1-C | share with A2 border share > 0 | 0.0 |  | n=5000 |
| framing residual | Kermany | E1-C | share with near-black edge run > 5% of side | 0.0026 |  | n=5000 |
| framing residual | Kermany | E1-C | share with near-black edge run > 10% of side | 0.0014 |  | n=5000 |
| framing residual | Kermany | E1-C | mean near-white share | 0.0205427867386895 |  | n=5000 |
| framing residual | MH | original | mean A2 border share | 0.42123297298027335 |  | n=31 |
| framing residual | MH | original | share with A2 border share > 0 | 1.0 |  | n=31 |
| framing residual | MH | original | share with near-black edge run > 5% of side | 1.0 |  | n=31 |
| framing residual | MH | original | share with near-black edge run > 10% of side | 0.967741935483871 |  | n=31 |
| framing residual | MH | original | mean near-white share | 0.0010593959466496318 |  | n=31 |
| framing residual | MH | E1-M | mean A2 border share | 0.017641129032258066 |  | n=31 |
| framing residual | MH | E1-M | share with A2 border share > 0 | 0.3225806451612903 |  | n=31 |
| framing residual | MH | E1-M | share with near-black edge run > 5% of side | 0.16129032258064516 |  | n=31 |
| framing residual | MH | E1-M | share with near-black edge run > 10% of side | 0.0967741935483871 |  | n=31 |
| framing residual | MH | E1-M | mean near-white share | 0.001136533675655242 |  | n=31 |
| framing residual | MH | E1-L | mean A2 border share | 0.023941532258064516 |  | n=31 |
| framing residual | MH | E1-L | share with A2 border share > 0 | 0.3870967741935484 |  | n=31 |
| framing residual | MH | E1-L | share with near-black edge run > 5% of side | 0.16129032258064516 |  | n=31 |
| framing residual | MH | E1-L | share with near-black edge run > 10% of side | 0.0967741935483871 |  | n=31 |
| framing residual | MH | E1-L | mean near-white share | 0.000762939453125 |  | n=31 |
| framing residual | MH | E1-C | mean A2 border share | 0.0 |  | n=31 |
| framing residual | MH | E1-C | share with A2 border share > 0 | 0.0 |  | n=31 |
| framing residual | MH | E1-C | share with near-black edge run > 5% of side | 0.12903225806451613 |  | n=31 |
| framing residual | MH | E1-C | share with near-black edge run > 10% of side | 0.06451612903225806 |  | n=31 |
| framing residual | MH | E1-C | mean near-white share | 0.0015050055744322997 |  | n=31 |
| framing residual | MacTel | original | mean A2 border share | 0.41274388162590175 |  | n=29 |
| framing residual | MacTel | original | share with A2 border share > 0 | 1.0 |  | n=29 |
| framing residual | MacTel | original | share with near-black edge run > 5% of side | 1.0 |  | n=29 |
| framing residual | MacTel | original | share with near-black edge run > 10% of side | 1.0 |  | n=29 |
| framing residual | MacTel | original | mean near-white share | 0.00025206060728471895 |  | n=29 |
| framing residual | MacTel | E1-M | mean A2 border share | 0.016298491379310345 |  | n=29 |
| framing residual | MacTel | E1-M | share with A2 border share > 0 | 0.1724137931034483 |  | n=29 |
| framing residual | MacTel | E1-M | share with near-black edge run > 5% of side | 0.27586206896551724 |  | n=29 |
| framing residual | MacTel | E1-M | share with near-black edge run > 10% of side | 0.2413793103448276 |  | n=29 |
| framing residual | MacTel | E1-M | mean near-white share | 0.00021151838631465518 |  | n=29 |
| framing residual | MacTel | E1-L | mean A2 border share | 0.029364224137931036 |  | n=29 |
| framing residual | MacTel | E1-L | share with A2 border share > 0 | 0.27586206896551724 |  | n=29 |
| framing residual | MacTel | E1-L | share with near-black edge run > 5% of side | 0.3103448275862069 |  | n=29 |
| framing residual | MacTel | E1-L | share with near-black edge run > 10% of side | 0.27586206896551724 |  | n=29 |
| framing residual | MacTel | E1-L | mean near-white share | 7.471544989224137e-05 |  | n=29 |
| framing residual | MacTel | E1-C | mean A2 border share | 0.0 |  | n=29 |
| framing residual | MacTel | E1-C | share with A2 border share > 0 | 0.0 |  | n=29 |
| framing residual | MacTel | E1-C | share with near-black edge run > 5% of side | 0.2413793103448276 |  | n=29 |
| framing residual | MacTel | E1-C | share with near-black edge run > 10% of side | 0.1724137931034483 |  | n=29 |
| framing residual | MacTel | E1-C | mean near-white share | 0.0004149157762177136 |  | n=29 |
| framing residual | OCTDL | original | mean A2 border share | 0.0 |  | n=371 |
| framing residual | OCTDL | original | share with A2 border share > 0 | 0.0 |  | n=371 |
| framing residual | OCTDL | original | share with near-black edge run > 5% of side | 0.0 |  | n=371 |
| framing residual | OCTDL | original | share with near-black edge run > 10% of side | 0.0 |  | n=371 |
| framing residual | OCTDL | original | mean near-white share | 0.0005733074268929488 |  | n=371 |
| framing residual | OCTDL | E1-M | mean A2 border share | 0.0 |  | n=371 |
| framing residual | OCTDL | E1-M | share with A2 border share > 0 | 0.0 |  | n=371 |
| framing residual | OCTDL | E1-M | share with near-black edge run > 5% of side | 0.0 |  | n=371 |
| framing residual | OCTDL | E1-M | share with near-black edge run > 10% of side | 0.0 |  | n=371 |
| framing residual | OCTDL | E1-M | mean near-white share | 0.0005768315811362871 |  | n=371 |
| framing residual | OCTDL | E1-L | mean A2 border share | 0.0006106805929919138 |  | n=371 |
| framing residual | OCTDL | E1-L | share with A2 border share > 0 | 0.02425876010781671 |  | n=371 |
| framing residual | OCTDL | E1-L | share with near-black edge run > 5% of side | 0.0 |  | n=371 |
| framing residual | OCTDL | E1-L | share with near-black edge run > 10% of side | 0.0 |  | n=371 |
| framing residual | OCTDL | E1-L | mean near-white share | 5.391987186236523e-05 |  | n=371 |
| framing residual | OCTDL | E1-C | mean A2 border share | 0.0 |  | n=371 |
| framing residual | OCTDL | E1-C | share with A2 border share > 0 | 0.0 |  | n=371 |
| framing residual | OCTDL | E1-C | share with near-black edge run > 5% of side | 0.0 |  | n=371 |
| framing residual | OCTDL | E1-C | share with near-black edge run > 10% of side | 0.0 |  | n=371 |
| framing residual | OCTDL | E1-C | mean near-white share | 0.0005733074268929488 |  | n=371 |
| framing residual | OCTID | original | mean A2 border share | 0.00034811629456787843 |  | n=410 |
| framing residual | OCTID | original | share with A2 border share > 0 | 0.06097560975609756 |  | n=410 |
| framing residual | OCTID | original | share with near-black edge run > 5% of side | 0.0 |  | n=410 |
| framing residual | OCTID | original | share with near-black edge run > 10% of side | 0.0 |  | n=410 |
| framing residual | OCTID | original | mean near-white share | 0.0026092129399066825 |  | n=410 |
| framing residual | OCTID | E1-M | mean A2 border share | 0.00031440548780487806 |  | n=410 |
| framing residual | OCTID | E1-M | share with A2 border share > 0 | 0.06097560975609756 |  | n=410 |
| framing residual | OCTID | E1-M | share with near-black edge run > 5% of side | 0.0 |  | n=410 |
| framing residual | OCTID | E1-M | share with near-black edge run > 10% of side | 0.0 |  | n=410 |
| framing residual | OCTID | E1-M | mean near-white share | 0.00026189292349466464 |  | n=410 |
| framing residual | OCTID | E1-L | mean A2 border share | 0.00016196646341463415 |  | n=410 |
| framing residual | OCTID | E1-L | share with A2 border share > 0 | 0.02195121951219512 |  | n=410 |
| framing residual | OCTID | E1-L | share with near-black edge run > 5% of side | 0.0 |  | n=410 |
| framing residual | OCTID | E1-L | share with near-black edge run > 10% of side | 0.0 |  | n=410 |
| framing residual | OCTID | E1-L | mean near-white share | 3.870522103658537e-06 |  | n=410 |
| framing residual | OCTID | E1-C | mean A2 border share | 0.0 |  | n=410 |
| framing residual | OCTID | E1-C | share with A2 border share > 0 | 0.0 |  | n=410 |
| framing residual | OCTID | E1-C | share with near-black edge run > 5% of side | 0.0 |  | n=410 |
| framing residual | OCTID | E1-C | share with near-black edge run > 10% of side | 0.0 |  | n=410 |
| framing residual | OCTID | E1-C | mean near-white share | 0.00260957197479844 |  | n=410 |
| framing residual | RP | original | mean A2 border share | 0.3006734521041314 |  | n=31 |
| framing residual | RP | original | share with A2 border share > 0 | 0.9032258064516129 |  | n=31 |
| framing residual | RP | original | share with near-black edge run > 5% of side | 0.8709677419354839 |  | n=31 |
| framing residual | RP | original | share with near-black edge run > 10% of side | 0.8709677419354839 |  | n=31 |
| framing residual | RP | original | mean near-white share | 0.00020969929657132627 |  | n=31 |
| framing residual | RP | E1-M | mean A2 border share | 0.02356350806451613 |  | n=31 |
| framing residual | RP | E1-M | share with A2 border share > 0 | 0.2903225806451613 |  | n=31 |
| framing residual | RP | E1-M | share with near-black edge run > 5% of side | 0.16129032258064516 |  | n=31 |
| framing residual | RP | E1-M | share with near-black edge run > 10% of side | 0.06451612903225806 |  | n=31 |
| framing residual | RP | E1-M | mean near-white share | 0.0002849948021673387 |  | n=31 |
| framing residual | RP | E1-L | mean A2 border share | 0.04259072580645161 |  | n=31 |
| framing residual | RP | E1-L | share with A2 border share > 0 | 0.41935483870967744 |  | n=31 |
| framing residual | RP | E1-L | share with near-black edge run > 5% of side | 0.1935483870967742 |  | n=31 |
| framing residual | RP | E1-L | share with near-black edge run > 10% of side | 0.06451612903225806 |  | n=31 |
| framing residual | RP | E1-L | mean near-white share | 0.00012551584551411291 |  | n=31 |
| framing residual | RP | E1-C | mean A2 border share | 0.0 |  | n=31 |
| framing residual | RP | E1-C | share with A2 border share > 0 | 0.0 |  | n=31 |
| framing residual | RP | E1-C | share with near-black edge run > 5% of side | 0.12903225806451613 |  | n=31 |
| framing residual | RP | E1-C | share with near-black edge run > 10% of side | 0.03225806451612903 |  | n=31 |
| framing residual | RP | E1-C | mean near-white share | 0.0003090760497227271 |  | n=31 |
| framing residual | Stargardt | original | mean A2 border share | 0.35319047496307715 |  | n=23 |
| framing residual | Stargardt | original | share with A2 border share > 0 | 0.9565217391304348 |  | n=23 |
| framing residual | Stargardt | original | share with near-black edge run > 5% of side | 0.9565217391304348 |  | n=23 |
| framing residual | Stargardt | original | share with near-black edge run > 10% of side | 0.9130434782608695 |  | n=23 |
| framing residual | Stargardt | original | mean near-white share | 8.831641243336315e-05 |  | n=23 |
| framing residual | Stargardt | E1-M | mean A2 border share | 0.031759510869565216 |  | n=23 |
| framing residual | Stargardt | E1-M | share with A2 border share > 0 | 0.391304347826087 |  | n=23 |
| framing residual | Stargardt | E1-M | share with near-black edge run > 5% of side | 0.08695652173913043 |  | n=23 |
| framing residual | Stargardt | E1-M | share with near-black edge run > 10% of side | 0.0 |  | n=23 |
| framing residual | Stargardt | E1-M | mean near-white share | 0.00012870456861413044 |  | n=23 |
| framing residual | Stargardt | E1-L | mean A2 border share | 0.04228940217391304 |  | n=23 |
| framing residual | Stargardt | E1-L | share with A2 border share > 0 | 0.5217391304347826 |  | n=23 |
| framing residual | Stargardt | E1-L | share with near-black edge run > 5% of side | 0.043478260869565216 |  | n=23 |
| framing residual | Stargardt | E1-L | share with near-black edge run > 10% of side | 0.0 |  | n=23 |
| framing residual | Stargardt | E1-L | mean near-white share | 3.0517578125e-05 |  | n=23 |
| framing residual | Stargardt | E1-C | mean A2 border share | 0.0 |  | n=23 |
| framing residual | Stargardt | E1-C | share with A2 border share > 0 | 0.0 |  | n=23 |
| framing residual | Stargardt | E1-C | share with near-black edge run > 5% of side | 0.21739130434782608 |  | n=23 |
| framing residual | Stargardt | E1-C | share with near-black edge run > 10% of side | 0.0 |  | n=23 |
| framing residual | Stargardt | E1-C | mean near-white share | 0.0001348262539549925 |  | n=23 |
| laptop re-render agreement |  | E1-M | images compared | 5146 |  | a report, never a stop |
| laptop re-render agreement |  | E1-M | images differing | 0 |  | a report, never a stop |
| laptop re-render agreement |  | E1-M | largest grey difference | 0 |  | a report, never a stop |
| laptop re-render agreement |  | E1-M | originals missing or differing from locked SHA256 | 781 |  | a report, never a stop |
| laptop re-render agreement |  | E1-M | originals not readable on this laptop | 0 |  | a report, never a stop |
| laptop re-render agreement |  | E1-L | images compared | 5146 |  | a report, never a stop |
| laptop re-render agreement |  | E1-L | images differing | 0 |  | a report, never a stop |
| laptop re-render agreement |  | E1-L | largest grey difference | 0 |  | a report, never a stop |
| laptop re-render agreement |  | E1-L | originals missing or differing from locked SHA256 | 781 |  | a report, never a stop |
| laptop re-render agreement |  | E1-L | originals not readable on this laptop | 0 |  | a report, never a stop |
| laptop re-render agreement |  | E1-C | images compared | 5146 |  | a report, never a stop |
| laptop re-render agreement |  | E1-C | images differing | 0 |  | a report, never a stop |
| laptop re-render agreement |  | E1-C | largest grey difference | 0 |  | a report, never a stop |
| laptop re-render agreement |  | E1-C | originals missing or differing from locked SHA256 | 781 |  | a report, never a stop |
| laptop re-render agreement |  | E1-C | originals not readable on this laptop | 0 |  | a report, never a stop |

## 3. Development, K mean (E1b; per K in the CSV)

| Model | Set | Status | Mean | SD | p05 | p95 |
|---|---|---|---|---|---|---|
| RETFound | locked | interpreted | 0.406 | 0.024 | 0.367 | 0.441 |
| RETFound | E1-M | interpreted | 0.412 | 0.029 | 0.364 | 0.458 |
| RETFound | E1-L | interpreted | 0.450 | 0.032 | 0.400 | 0.501 |
| RETFound | E1-C | interpreted | 0.412 | 0.029 | 0.363 | 0.456 |
| DINOv2 | locked | interpreted | 0.558 | 0.028 | 0.511 | 0.606 |
| DINOv2 | E1-M | interpreted | 0.542 | 0.028 | 0.499 | 0.587 |
| DINOv2 | E1-L | interpreted | 0.576 | 0.030 | 0.529 | 0.622 |
| DINOv2 | E1-C | interpreted | 0.540 | 0.028 | 0.498 | 0.588 |
| MIRAGE | locked | interpreted | 0.650 | 0.028 | 0.603 | 0.688 |
| MIRAGE | E1-M | interpreted | 0.649 | 0.024 | 0.611 | 0.692 |
| MIRAGE | E1-L | interpreted | 0.655 | 0.024 | 0.618 | 0.694 |
| MIRAGE | E1-C | interpreted | 0.622 | 0.024 | 0.585 | 0.660 |
| Inception | locked | interpreted | 0.534 | 0.029 | 0.494 | 0.583 |
| Inception | E1-M | interpreted | 0.586 | 0.027 | 0.540 | 0.635 |
| Inception | E1-L | interpreted | 0.578 | 0.025 | 0.537 | 0.614 |
| Inception | E1-C | interpreted | 0.555 | 0.031 | 0.510 | 0.608 |
| size_only | raw | reference without a model | 0.569 | 0.036 | 0.508 | 0.619 |
| content_size | raw content box | reference without a model | 0.447 | 0.028 | 0.395 | 0.495 |
| thumbnail | raw | reference without a model | 0.288 | 0.025 | 0.246 | 0.325 |
| thumbnail | E1-M | reference without a model | 0.360 | 0.025 | 0.325 | 0.399 |
| thumbnail | E1-L | reference without a model | 0.360 | 0.025 | 0.325 | 0.399 |
| thumbnail | E1-C | reference without a model | 0.331 | 0.024 | 0.292 | 0.364 |

## 4. Paired differences, K mean (E1c)

| Comparison | Set | Status | Mean | p05 | p95 |
|---|---|---|---|---|---|
| RETFound minus Inception | locked | interpreted | -0.128 | -0.186 | -0.070 |
| DINOv2 minus Inception | locked | interpreted | 0.024 | -0.025 | 0.086 |
| MIRAGE minus Inception | locked | interpreted | 0.116 | 0.069 | 0.185 |
| RETFound minus Inception | E1-M | interpreted | -0.175 | -0.236 | -0.124 |
| DINOv2 minus Inception | E1-M | interpreted | -0.044 | -0.099 | 0.004 |
| MIRAGE minus Inception | E1-M | interpreted | 0.063 | 0.003 | 0.117 |
| RETFound minus Inception | E1-L | interpreted | -0.128 | -0.187 | -0.071 |
| DINOv2 minus Inception | E1-L | interpreted | -0.002 | -0.054 | 0.051 |
| MIRAGE minus Inception | E1-L | interpreted | 0.077 | 0.023 | 0.132 |
| RETFound minus Inception | E1-C | interpreted | -0.143 | -0.201 | -0.085 |
| DINOv2 minus Inception | E1-C | interpreted | -0.014 | -0.072 | 0.052 |
| MIRAGE minus Inception | E1-C | interpreted | 0.068 | 0.017 | 0.123 |
| RETFound E1-M minus RETFound locked | E1-M | interpreted | 0.005 | -0.037 | 0.051 |
| DINOv2 E1-M minus DINOv2 locked | E1-M | interpreted | -0.015 | -0.058 | 0.034 |
| MIRAGE E1-M minus MIRAGE locked | E1-M | interpreted | -0.000 | -0.051 | 0.043 |
| Inception E1-M minus Inception locked | E1-M | interpreted | 0.052 | 0.006 | 0.093 |
| RETFound E1-L minus RETFound locked | E1-L | interpreted | 0.044 | -0.000 | 0.100 |
| DINOv2 E1-L minus DINOv2 locked | E1-L | interpreted | 0.018 | -0.025 | 0.064 |
| MIRAGE E1-L minus MIRAGE locked | E1-L | interpreted | 0.005 | -0.046 | 0.048 |
| Inception E1-L minus Inception locked | E1-L | interpreted | 0.044 | -0.004 | 0.093 |
| RETFound E1-C minus RETFound locked | E1-C | interpreted | 0.006 | -0.033 | 0.052 |
| DINOv2 E1-C minus DINOv2 locked | E1-C | interpreted | -0.018 | -0.066 | 0.040 |
| MIRAGE E1-C minus MIRAGE locked | E1-C | interpreted | -0.027 | -0.077 | 0.021 |
| Inception E1-C minus Inception locked | E1-C | interpreted | 0.021 | -0.018 | 0.057 |

## 5. Matched subset (E1e; supports are not matched)

| Model | Set | k | n queries | Classes | Mean | p05 |
|---|---|---|---|---|---|---|
| matched subset | L=164 px, U=296 px |  | 92 |  |  |  |
| RETFound | locked | mean_k1_2_4_8 | 92 | CSC MH MacTel Stargardt RP | 0.408 | 0.361 |
| RETFound | E1-M | mean_k1_2_4_8 | 92 | CSC MH MacTel Stargardt RP | 0.417 | 0.364 |
| RETFound | E1-L | mean_k1_2_4_8 | 92 | CSC MH MacTel Stargardt RP | 0.452 | 0.400 |
| RETFound | E1-C | mean_k1_2_4_8 | 92 | CSC MH MacTel Stargardt RP | 0.417 | 0.358 |
| DINOv2 | locked | mean_k1_2_4_8 | 92 | CSC MH MacTel Stargardt RP | 0.556 | 0.511 |
| DINOv2 | E1-M | mean_k1_2_4_8 | 92 | CSC MH MacTel Stargardt RP | 0.534 | 0.478 |
| DINOv2 | E1-L | mean_k1_2_4_8 | 92 | CSC MH MacTel Stargardt RP | 0.572 | 0.517 |
| DINOv2 | E1-C | mean_k1_2_4_8 | 92 | CSC MH MacTel Stargardt RP | 0.530 | 0.477 |
| MIRAGE | locked | mean_k1_2_4_8 | 92 | CSC MH MacTel Stargardt RP | 0.653 | 0.599 |
| MIRAGE | E1-M | mean_k1_2_4_8 | 92 | CSC MH MacTel Stargardt RP | 0.657 | 0.606 |
| MIRAGE | E1-L | mean_k1_2_4_8 | 92 | CSC MH MacTel Stargardt RP | 0.657 | 0.616 |
| MIRAGE | E1-C | mean_k1_2_4_8 | 92 | CSC MH MacTel Stargardt RP | 0.645 | 0.594 |
| Inception | locked | mean_k1_2_4_8 | 92 | CSC MH MacTel Stargardt RP | 0.526 | 0.478 |
| Inception | E1-M | mean_k1_2_4_8 | 92 | CSC MH MacTel Stargardt RP | 0.600 | 0.550 |
| Inception | E1-L | mean_k1_2_4_8 | 92 | CSC MH MacTel Stargardt RP | 0.589 | 0.538 |
| Inception | E1-C | mean_k1_2_4_8 | 92 | CSC MH MacTel Stargardt RP | 0.561 | 0.514 |
| MIRAGE minus Inception | locked | mean_k1_2_4_8 | 92 |  | 0.127 | 0.067 |
| MIRAGE minus Inception | E1-M | mean_k1_2_4_8 | 92 |  | 0.057 | -0.012 |
| MIRAGE minus Inception | E1-L | mean_k1_2_4_8 | 92 |  | 0.068 | 0.010 |
| MIRAGE minus Inception | E1-C | mean_k1_2_4_8 | 92 |  | 0.084 | 0.027 |

## 6. Hold-out, legacy 22 (E1f; direction only, K = 16 shown and never read)

| Comparison | Set | k | Mean paired difference | Direction |
|---|---|---|---|---|
| MIRAGE minus Inception | locked | 1 | 0.106 | positive |
| MIRAGE minus Inception | locked | 2 | 0.188 | positive |
| MIRAGE minus Inception | locked | 4 | 0.243 | positive |
| MIRAGE minus Inception | locked | 8 | 0.253 | positive |
| MIRAGE minus Inception | locked | 16 | 0.273 |  |
| MIRAGE minus Inception | locked | mean_k1_2_4_8 | 0.197 | positive |
| MIRAGE minus Inception | E1-M | 1 | 0.086 | positive |
| MIRAGE minus Inception | E1-M | 2 | 0.101 | positive |
| MIRAGE minus Inception | E1-M | 4 | 0.092 | positive |
| MIRAGE minus Inception | E1-M | 8 | 0.106 | positive |
| MIRAGE minus Inception | E1-M | 16 | 0.131 |  |
| MIRAGE minus Inception | E1-M | mean_k1_2_4_8 | 0.096 | positive |
| MIRAGE minus Inception | E1-L | 1 | 0.117 | positive |
| MIRAGE minus Inception | E1-L | 2 | 0.124 | positive |
| MIRAGE minus Inception | E1-L | 4 | 0.137 | positive |
| MIRAGE minus Inception | E1-L | 8 | 0.131 | positive |
| MIRAGE minus Inception | E1-L | 16 | 0.111 |  |
| MIRAGE minus Inception | E1-L | mean_k1_2_4_8 | 0.127 | positive |
| MIRAGE minus Inception | E1-C | 1 | 0.081 | positive |
| MIRAGE minus Inception | E1-C | 2 | 0.102 | positive |
| MIRAGE minus Inception | E1-C | 4 | 0.111 | positive |
| MIRAGE minus Inception | E1-C | 8 | 0.135 | positive |
| MIRAGE minus Inception | E1-C | 16 | 0.161 |  |
| MIRAGE minus Inception | E1-C | mean_k1_2_4_8 | 0.107 | positive |

## 7. Nine-way audit (E1g)

| Model | Set | Status | Query set | n | Correct | NORMAL | CNV, DME or DRUSEN | Any rare | NORMAL recall | Rate | 95% (locked method) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| RETFound | locked | interpreted | Kermany patient-disjoint | 169 | 149 | 149 | 19 | 1 | 0.882 | false_rare_rate 0.006 | 0.000 to 0.019 |
| RETFound | locked | interpreted | Legacy 22 | 22 | 9 | 0 | 3 | 19 |  |   |  |
| RETFound | locked | interpreted | OCTID NORMAL | 206 | 189 | 189 | 17 | 0 | 0.917 | false_rare_rate 0.000 | 0.000 to 0.018 |
| RETFound | locked | interpreted | OCTID CSC | 102 | 0 | 13 | 89 | 0 |  | share_called_any_rare (R1) 0.000 |  |
| RETFound | locked | interpreted | OCTID MH | 102 | 0 | 8 | 94 | 0 |  | share_called_any_rare (R1) 0.000 |  |
| RETFound | locked | interpreted | OCTDL NORMAL | 332 | 137 | 137 | 177 | 18 | 0.413 | false_rare_rate 0.054 | 0.029 to 0.087 |
| RETFound | locked | interpreted | OCTDL MH | 39 | 0 | 0 | 36 | 3 |  | share_called_any_rare (R1) 0.077 |  |
| RETFound | E1-M | interpreted | Kermany patient-disjoint | 169 | 149 | 149 | 19 | 1 | 0.882 | false_rare_rate 0.006 | 0.000 to 0.019 |
| RETFound | E1-M | interpreted | Legacy 22 | 22 | 10 | 0 | 2 | 20 |  |   |  |
| RETFound | E1-M | interpreted | OCTID NORMAL | 206 | 198 | 198 | 8 | 0 | 0.961 | false_rare_rate 0.000 | 0.000 to 0.018 |
| RETFound | E1-M | interpreted | OCTID CSC | 102 | 0 | 12 | 88 | 2 |  | share_called_any_rare (R1) 0.020 |  |
| RETFound | E1-M | interpreted | OCTID MH | 102 | 0 | 9 | 93 | 0 |  | share_called_any_rare (R1) 0.000 |  |
| RETFound | E1-M | interpreted | OCTDL NORMAL | 332 | 92 | 92 | 109 | 131 | 0.277 | false_rare_rate 0.395 | 0.326 to 0.467 |
| RETFound | E1-M | interpreted | OCTDL MH | 39 | 3 | 0 | 23 | 16 |  | share_called_any_rare (R1) 0.410 |  |
| RETFound | E1-L | interpreted | Kermany patient-disjoint | 169 | 154 | 154 | 15 | 0 | 0.911 | false_rare_rate 0.000 | 0.000 to 0.000 |
| RETFound | E1-L | interpreted | Legacy 22 | 22 | 15 | 1 | 0 | 21 |  |   |  |
| RETFound | E1-L | interpreted | OCTID NORMAL | 206 | 157 | 157 | 8 | 41 | 0.762 | false_rare_rate 0.199 | 0.147 to 0.260 |
| RETFound | E1-L | interpreted | OCTID CSC | 102 | 0 | 36 | 36 | 30 |  | share_called_any_rare (R1) 0.294 |  |
| RETFound | E1-L | interpreted | OCTID MH | 102 | 42 | 17 | 39 | 46 |  | share_called_any_rare (R1) 0.451 |  |
| RETFound | E1-L | interpreted | OCTDL NORMAL | 332 | 31 | 31 | 2 | 299 | 0.093 | false_rare_rate 0.901 | 0.862 to 0.930 |
| RETFound | E1-L | interpreted | OCTDL MH | 39 | 34 | 0 | 0 | 39 |  | share_called_any_rare (R1) 1.000 |  |
| RETFound | E1-C | interpreted | Kermany patient-disjoint | 169 | 149 | 149 | 19 | 1 | 0.882 | false_rare_rate 0.006 | 0.000 to 0.019 |
| RETFound | E1-C | interpreted | Legacy 22 | 22 | 10 | 0 | 2 | 20 |  |   |  |
| RETFound | E1-C | interpreted | OCTID NORMAL | 206 | 199 | 199 | 7 | 0 | 0.966 | false_rare_rate 0.000 | 0.000 to 0.018 |
| RETFound | E1-C | interpreted | OCTID CSC | 102 | 0 | 12 | 88 | 2 |  | share_called_any_rare (R1) 0.020 |  |
| RETFound | E1-C | interpreted | OCTID MH | 102 | 0 | 9 | 93 | 0 |  | share_called_any_rare (R1) 0.000 |  |
| RETFound | E1-C | interpreted | OCTDL NORMAL | 332 | 92 | 92 | 112 | 128 | 0.277 | false_rare_rate 0.386 | 0.316 to 0.458 |
| RETFound | E1-C | interpreted | OCTDL MH | 39 | 3 | 0 | 23 | 16 |  | share_called_any_rare (R1) 0.410 |  |
| DINOv2 | locked | interpreted | Kermany patient-disjoint | 169 | 146 | 146 | 17 | 6 | 0.864 | false_rare_rate 0.036 | 0.012 to 0.065 |
| DINOv2 | locked | interpreted | Legacy 22 | 22 | 15 | 0 | 1 | 21 |  |   |  |
| DINOv2 | locked | interpreted | OCTID NORMAL | 206 | 188 | 188 | 1 | 17 | 0.913 | false_rare_rate 0.083 | 0.049 to 0.129 |
| DINOv2 | locked | interpreted | OCTID CSC | 102 | 30 | 11 | 22 | 69 |  | share_called_any_rare (R1) 0.676 |  |
| DINOv2 | locked | interpreted | OCTID MH | 102 | 48 | 4 | 13 | 85 |  | share_called_any_rare (R1) 0.833 |  |
| DINOv2 | locked | interpreted | OCTDL NORMAL | 332 | 145 | 145 | 46 | 141 | 0.437 | false_rare_rate 0.425 | 0.372 to 0.472 |
| DINOv2 | locked | interpreted | OCTDL MH | 39 | 26 | 0 | 0 | 39 |  | share_called_any_rare (R1) 1.000 |  |
| DINOv2 | E1-M | interpreted | Kermany patient-disjoint | 169 | 153 | 153 | 15 | 1 | 0.905 | false_rare_rate 0.006 | 0.000 to 0.018 |
| DINOv2 | E1-M | interpreted | Legacy 22 | 22 | 15 | 0 | 2 | 20 |  |   |  |
| DINOv2 | E1-M | interpreted | OCTID NORMAL | 206 | 203 | 203 | 1 | 2 | 0.985 | false_rare_rate 0.010 | 0.001 to 0.035 |
| DINOv2 | E1-M | interpreted | OCTID CSC | 102 | 12 | 21 | 49 | 32 |  | share_called_any_rare (R1) 0.314 |  |
| DINOv2 | E1-M | interpreted | OCTID MH | 102 | 36 | 7 | 23 | 72 |  | share_called_any_rare (R1) 0.706 |  |
| DINOv2 | E1-M | interpreted | OCTDL NORMAL | 332 | 162 | 162 | 90 | 80 | 0.488 | false_rare_rate 0.241 | 0.181 to 0.297 |
| DINOv2 | E1-M | interpreted | OCTDL MH | 39 | 27 | 0 | 2 | 37 |  | share_called_any_rare (R1) 0.949 |  |
| DINOv2 | E1-L | interpreted | Kermany patient-disjoint | 169 | 149 | 149 | 20 | 0 | 0.882 | false_rare_rate 0.000 | 0.000 to 0.000 |
| DINOv2 | E1-L | interpreted | Legacy 22 | 22 | 12 | 1 | 2 | 19 |  |   |  |
| DINOv2 | E1-L | interpreted | OCTID NORMAL | 206 | 56 | 56 | 0 | 150 | 0.272 | false_rare_rate 0.728 | 0.662 to 0.788 |
| DINOv2 | E1-L | interpreted | OCTID CSC | 102 | 28 | 3 | 16 | 83 |  | share_called_any_rare (R1) 0.814 |  |
| DINOv2 | E1-L | interpreted | OCTID MH | 102 | 50 | 1 | 12 | 89 |  | share_called_any_rare (R1) 0.873 |  |
| DINOv2 | E1-L | interpreted | OCTDL NORMAL | 332 | 128 | 128 | 97 | 107 | 0.386 | false_rare_rate 0.322 | 0.268 to 0.382 |
| DINOv2 | E1-L | interpreted | OCTDL MH | 39 | 29 | 1 | 2 | 36 |  | share_called_any_rare (R1) 0.923 |  |
| DINOv2 | E1-C | interpreted | Kermany patient-disjoint | 169 | 152 | 152 | 16 | 1 | 0.899 | false_rare_rate 0.006 | 0.000 to 0.018 |
| DINOv2 | E1-C | interpreted | Legacy 22 | 22 | 14 | 0 | 2 | 20 |  |   |  |
| DINOv2 | E1-C | interpreted | OCTID NORMAL | 206 | 203 | 203 | 1 | 2 | 0.985 | false_rare_rate 0.010 | 0.001 to 0.035 |
| DINOv2 | E1-C | interpreted | OCTID CSC | 102 | 11 | 21 | 50 | 31 |  | share_called_any_rare (R1) 0.304 |  |
| DINOv2 | E1-C | interpreted | OCTID MH | 102 | 32 | 8 | 23 | 71 |  | share_called_any_rare (R1) 0.696 |  |
| DINOv2 | E1-C | interpreted | OCTDL NORMAL | 332 | 180 | 180 | 89 | 63 | 0.542 | false_rare_rate 0.190 | 0.142 to 0.233 |
| DINOv2 | E1-C | interpreted | OCTDL MH | 39 | 27 | 0 | 2 | 37 |  | share_called_any_rare (R1) 0.949 |  |
| MIRAGE | locked | interpreted | Kermany patient-disjoint | 169 | 169 | 169 | 0 | 0 | 1.000 | false_rare_rate 0.000 | 0.000 to 0.000 |
| MIRAGE | locked | interpreted | Legacy 22 | 22 | 20 | 0 | 0 | 22 |  |   |  |
| MIRAGE | locked | interpreted | OCTID NORMAL | 206 | 202 | 202 | 0 | 4 | 0.981 | false_rare_rate 0.019 | 0.005 to 0.049 |
| MIRAGE | locked | interpreted | OCTID CSC | 102 | 23 | 8 | 53 | 41 |  | share_called_any_rare (R1) 0.402 |  |
| MIRAGE | locked | interpreted | OCTID MH | 102 | 10 | 8 | 83 | 11 |  | share_called_any_rare (R1) 0.108 |  |
| MIRAGE | locked | interpreted | OCTDL NORMAL | 332 | 26 | 26 | 276 | 30 | 0.078 | false_rare_rate 0.090 | 0.055 to 0.136 |
| MIRAGE | locked | interpreted | OCTDL MH | 39 | 3 | 0 | 36 | 3 |  | share_called_any_rare (R1) 0.077 |  |
| MIRAGE | E1-M | interpreted | Kermany patient-disjoint | 169 | 169 | 169 | 0 | 0 | 1.000 | false_rare_rate 0.000 | 0.000 to 0.000 |
| MIRAGE | E1-M | interpreted | Legacy 22 | 22 | 18 | 0 | 1 | 21 |  |   |  |
| MIRAGE | E1-M | interpreted | OCTID NORMAL | 206 | 206 | 206 | 0 | 0 | 1.000 | false_rare_rate 0.000 | 0.000 to 0.018 |
| MIRAGE | E1-M | interpreted | OCTID CSC | 102 | 6 | 17 | 51 | 34 |  | share_called_any_rare (R1) 0.333 |  |
| MIRAGE | E1-M | interpreted | OCTID MH | 102 | 11 | 8 | 81 | 13 |  | share_called_any_rare (R1) 0.127 |  |
| MIRAGE | E1-M | interpreted | OCTDL NORMAL | 332 | 65 | 65 | 6 | 261 | 0.196 | false_rare_rate 0.786 | 0.719 to 0.838 |
| MIRAGE | E1-M | interpreted | OCTDL MH | 39 | 36 | 0 | 0 | 39 |  | share_called_any_rare (R1) 1.000 |  |
| MIRAGE | E1-L | interpreted | Kermany patient-disjoint | 169 | 165 | 165 | 4 | 0 | 0.976 | false_rare_rate 0.000 | 0.000 to 0.000 |
| MIRAGE | E1-L | interpreted | Legacy 22 | 22 | 18 | 0 | 1 | 21 |  |   |  |
| MIRAGE | E1-L | interpreted | OCTID NORMAL | 206 | 206 | 206 | 0 | 0 | 1.000 | false_rare_rate 0.000 | 0.000 to 0.018 |
| MIRAGE | E1-L | interpreted | OCTID CSC | 102 | 8 | 23 | 49 | 30 |  | share_called_any_rare (R1) 0.294 |  |
| MIRAGE | E1-L | interpreted | OCTID MH | 102 | 12 | 13 | 75 | 14 |  | share_called_any_rare (R1) 0.137 |  |
| MIRAGE | E1-L | interpreted | OCTDL NORMAL | 332 | 38 | 38 | 3 | 291 | 0.114 | false_rare_rate 0.877 | 0.819 to 0.920 |
| MIRAGE | E1-L | interpreted | OCTDL MH | 39 | 36 | 0 | 0 | 39 |  | share_called_any_rare (R1) 1.000 |  |
| MIRAGE | E1-C | interpreted | Kermany patient-disjoint | 169 | 169 | 169 | 0 | 0 | 1.000 | false_rare_rate 0.000 | 0.000 to 0.000 |
| MIRAGE | E1-C | interpreted | Legacy 22 | 22 | 18 | 0 | 2 | 20 |  |   |  |
| MIRAGE | E1-C | interpreted | OCTID NORMAL | 206 | 206 | 206 | 0 | 0 | 1.000 | false_rare_rate 0.000 | 0.000 to 0.018 |
| MIRAGE | E1-C | interpreted | OCTID CSC | 102 | 3 | 12 | 76 | 14 |  | share_called_any_rare (R1) 0.137 |  |
| MIRAGE | E1-C | interpreted | OCTID MH | 102 | 2 | 10 | 89 | 3 |  | share_called_any_rare (R1) 0.029 |  |
| MIRAGE | E1-C | interpreted | OCTDL NORMAL | 332 | 29 | 29 | 75 | 228 | 0.087 | false_rare_rate 0.687 | 0.613 to 0.750 |
| MIRAGE | E1-C | interpreted | OCTDL MH | 39 | 30 | 0 | 9 | 30 |  | share_called_any_rare (R1) 0.769 |  |
| Inception | locked | interpreted | Kermany patient-disjoint | 169 | 150 | 150 | 15 | 4 | 0.888 | false_rare_rate 0.024 | 0.006 to 0.049 |
| Inception | locked | interpreted | Legacy 22 | 22 | 13 | 0 | 2 | 20 |  |   |  |
| Inception | locked | interpreted | OCTID NORMAL | 206 | 182 | 182 | 9 | 15 | 0.883 | false_rare_rate 0.073 | 0.041 to 0.117 |
| Inception | locked | interpreted | OCTID CSC | 102 | 22 | 3 | 53 | 46 |  | share_called_any_rare (R1) 0.451 |  |
| Inception | locked | interpreted | OCTID MH | 102 | 27 | 6 | 52 | 44 |  | share_called_any_rare (R1) 0.431 |  |
| Inception | locked | interpreted | OCTDL NORMAL | 332 | 66 | 66 | 255 | 11 | 0.199 | false_rare_rate 0.033 | 0.012 to 0.059 |
| Inception | locked | interpreted | OCTDL MH | 39 | 0 | 2 | 36 | 1 |  | share_called_any_rare (R1) 0.026 |  |
| Inception | E1-M | interpreted | Kermany patient-disjoint | 169 | 160 | 160 | 7 | 2 | 0.947 | false_rare_rate 0.012 | 0.000 to 0.030 |
| Inception | E1-M | interpreted | Legacy 22 | 22 | 16 | 0 | 4 | 18 |  |   |  |
| Inception | E1-M | interpreted | OCTID NORMAL | 206 | 125 | 125 | 5 | 76 | 0.607 | false_rare_rate 0.369 | 0.303 to 0.439 |
| Inception | E1-M | interpreted | OCTID CSC | 102 | 14 | 4 | 60 | 38 |  | share_called_any_rare (R1) 0.373 |  |
| Inception | E1-M | interpreted | OCTID MH | 102 | 16 | 4 | 75 | 23 |  | share_called_any_rare (R1) 0.225 |  |
| Inception | E1-M | interpreted | OCTDL NORMAL | 332 | 115 | 115 | 123 | 94 | 0.346 | false_rare_rate 0.283 | 0.227 to 0.336 |
| Inception | E1-M | interpreted | OCTDL MH | 39 | 29 | 0 | 9 | 30 |  | share_called_any_rare (R1) 0.769 |  |
| Inception | E1-L | interpreted | Kermany patient-disjoint | 169 | 155 | 155 | 12 | 2 | 0.917 | false_rare_rate 0.012 | 0.000 to 0.030 |
| Inception | E1-L | interpreted | Legacy 22 | 22 | 14 | 0 | 4 | 18 |  |   |  |
| Inception | E1-L | interpreted | OCTID NORMAL | 206 | 123 | 123 | 15 | 68 | 0.597 | false_rare_rate 0.330 | 0.266 to 0.399 |
| Inception | E1-L | interpreted | OCTID CSC | 102 | 16 | 4 | 57 | 41 |  | share_called_any_rare (R1) 0.402 |  |
| Inception | E1-L | interpreted | OCTID MH | 102 | 19 | 5 | 67 | 30 |  | share_called_any_rare (R1) 0.294 |  |
| Inception | E1-L | interpreted | OCTDL NORMAL | 332 | 152 | 152 | 110 | 70 | 0.458 | false_rare_rate 0.211 | 0.163 to 0.265 |
| Inception | E1-L | interpreted | OCTDL MH | 39 | 31 | 0 | 8 | 31 |  | share_called_any_rare (R1) 0.795 |  |
| Inception | E1-C | interpreted | Kermany patient-disjoint | 169 | 148 | 148 | 17 | 4 | 0.876 | false_rare_rate 0.024 | 0.006 to 0.048 |
| Inception | E1-C | interpreted | Legacy 22 | 22 | 15 | 1 | 4 | 17 |  |   |  |
| Inception | E1-C | interpreted | OCTID NORMAL | 206 | 146 | 146 | 9 | 51 | 0.709 | false_rare_rate 0.248 | 0.190 to 0.312 |
| Inception | E1-C | interpreted | OCTID CSC | 102 | 15 | 6 | 49 | 47 |  | share_called_any_rare (R1) 0.461 |  |
| Inception | E1-C | interpreted | OCTID MH | 102 | 24 | 5 | 52 | 45 |  | share_called_any_rare (R1) 0.441 |  |
| Inception | E1-C | interpreted | OCTDL NORMAL | 332 | 34 | 34 | 135 | 163 | 0.102 | false_rare_rate 0.491 | 0.424 to 0.549 |
| Inception | E1-C | interpreted | OCTDL MH | 39 | 23 | 0 | 7 | 32 |  | share_called_any_rare (R1) 0.821 |  |
| size_only | raw | reference without a model | Kermany patient-disjoint | 169 | 0 | 0 | 169 | 0 | 0.000 | false_rare_rate 0.000 | 0.000 to 0.000 |
| size_only | raw | reference without a model | Legacy 22 | 22 | 17 | 0 | 0 | 22 |  |   |  |
| size_only | raw | reference without a model | OCTID NORMAL | 206 | 0 | 0 | 206 | 0 | 0.000 | false_rare_rate 0.000 | 0.000 to 0.018 |
| size_only | raw | reference without a model | OCTID CSC | 102 | 0 | 0 | 102 | 0 |  | share_called_any_rare (R1) 0.000 |  |
| size_only | raw | reference without a model | OCTID MH | 102 | 0 | 0 | 102 | 0 |  | share_called_any_rare (R1) 0.000 |  |
| size_only | raw | reference without a model | OCTDL NORMAL | 332 | 0 | 0 | 323 | 9 | 0.000 | false_rare_rate 0.027 | 0.007 to 0.055 |
| size_only | raw | reference without a model | OCTDL MH | 39 | 0 | 0 | 39 | 0 |  | share_called_any_rare (R1) 0.000 |  |
| content_size | raw content box | reference without a model | Kermany patient-disjoint | 169 | 0 | 0 | 169 | 0 | 0.000 | false_rare_rate 0.000 | 0.000 to 0.000 |
| content_size | raw content box | reference without a model | Legacy 22 | 22 | 12 | 0 | 0 | 22 |  |   |  |
| content_size | raw content box | reference without a model | OCTID NORMAL | 206 | 0 | 0 | 206 | 0 | 0.000 | false_rare_rate 0.000 | 0.000 to 0.018 |
| content_size | raw content box | reference without a model | OCTID CSC | 102 | 0 | 0 | 102 | 0 |  | share_called_any_rare (R1) 0.000 |  |
| content_size | raw content box | reference without a model | OCTID MH | 102 | 0 | 0 | 102 | 0 |  | share_called_any_rare (R1) 0.000 |  |
| content_size | raw content box | reference without a model | OCTDL NORMAL | 332 | 0 | 0 | 323 | 9 | 0.000 | false_rare_rate 0.027 | 0.007 to 0.055 |
| content_size | raw content box | reference without a model | OCTDL MH | 39 | 0 | 0 | 39 | 0 |  | share_called_any_rare (R1) 0.000 |  |
| thumbnail | raw | reference without a model | Kermany patient-disjoint | 169 | 87 | 87 | 69 | 13 | 0.515 | false_rare_rate 0.077 | 0.037 to 0.121 |
| thumbnail | raw | reference without a model | Legacy 22 | 22 | 2 | 1 | 1 | 20 |  |   |  |
| thumbnail | raw | reference without a model | OCTID NORMAL | 206 | 0 | 0 | 203 | 3 | 0.000 | false_rare_rate 0.015 | 0.003 to 0.042 |
| thumbnail | raw | reference without a model | OCTID CSC | 102 | 1 | 5 | 81 | 16 |  | share_called_any_rare (R1) 0.157 |  |
| thumbnail | raw | reference without a model | OCTID MH | 102 | 2 | 17 | 77 | 8 |  | share_called_any_rare (R1) 0.078 |  |
| thumbnail | raw | reference without a model | OCTDL NORMAL | 332 | 4 | 4 | 69 | 259 | 0.012 | false_rare_rate 0.780 | 0.721 to 0.831 |
| thumbnail | raw | reference without a model | OCTDL MH | 39 | 5 | 0 | 12 | 27 |  | share_called_any_rare (R1) 0.692 |  |
| thumbnail | E1-M | reference without a model | Kermany patient-disjoint | 169 | 84 | 84 | 73 | 12 | 0.497 | false_rare_rate 0.071 | 0.036 to 0.112 |
| thumbnail | E1-M | reference without a model | Legacy 22 | 22 | 5 | 2 | 2 | 18 |  |   |  |
| thumbnail | E1-M | reference without a model | OCTID NORMAL | 206 | 0 | 0 | 42 | 164 | 0.000 | false_rare_rate 0.796 | 0.735 to 0.849 |
| thumbnail | E1-M | reference without a model | OCTID CSC | 102 | 28 | 6 | 41 | 55 |  | share_called_any_rare (R1) 0.539 |  |
| thumbnail | E1-M | reference without a model | OCTID MH | 102 | 12 | 15 | 47 | 40 |  | share_called_any_rare (R1) 0.392 |  |
| thumbnail | E1-M | reference without a model | OCTDL NORMAL | 332 | 1 | 1 | 0 | 331 | 0.003 | false_rare_rate 0.997 | 0.990 to 1.000 |
| thumbnail | E1-M | reference without a model | OCTDL MH | 39 | 17 | 0 | 0 | 39 |  | share_called_any_rare (R1) 1.000 |  |
| thumbnail | E1-L | reference without a model | Kermany patient-disjoint | 169 | 84 | 84 | 73 | 12 | 0.497 | false_rare_rate 0.071 | 0.036 to 0.112 |
| thumbnail | E1-L | reference without a model | Legacy 22 | 22 | 6 | 2 | 1 | 19 |  |   |  |
| thumbnail | E1-L | reference without a model | OCTID NORMAL | 206 | 0 | 0 | 40 | 166 | 0.000 | false_rare_rate 0.806 | 0.745 to 0.858 |
| thumbnail | E1-L | reference without a model | OCTID CSC | 102 | 29 | 6 | 40 | 56 |  | share_called_any_rare (R1) 0.549 |  |
| thumbnail | E1-L | reference without a model | OCTID MH | 102 | 12 | 15 | 48 | 39 |  | share_called_any_rare (R1) 0.382 |  |
| thumbnail | E1-L | reference without a model | OCTDL NORMAL | 332 | 1 | 1 | 0 | 331 | 0.003 | false_rare_rate 0.997 | 0.990 to 1.000 |
| thumbnail | E1-L | reference without a model | OCTDL MH | 39 | 17 | 0 | 0 | 39 |  | share_called_any_rare (R1) 1.000 |  |
| thumbnail | E1-C | reference without a model | Kermany patient-disjoint | 169 | 86 | 86 | 69 | 14 | 0.509 | false_rare_rate 0.083 | 0.043 to 0.126 |
| thumbnail | E1-C | reference without a model | Legacy 22 | 22 | 7 | 2 | 0 | 20 |  |   |  |
| thumbnail | E1-C | reference without a model | OCTID NORMAL | 206 | 0 | 0 | 136 | 70 | 0.000 | false_rare_rate 0.340 | 0.275 to 0.409 |
| thumbnail | E1-C | reference without a model | OCTID CSC | 102 | 5 | 6 | 59 | 37 |  | share_called_any_rare (R1) 0.363 |  |
| thumbnail | E1-C | reference without a model | OCTID MH | 102 | 11 | 19 | 60 | 23 |  | share_called_any_rare (R1) 0.225 |  |
| thumbnail | E1-C | reference without a model | OCTDL NORMAL | 332 | 1 | 1 | 3 | 328 | 0.003 | false_rare_rate 0.988 | 0.976 to 0.997 |
| thumbnail | E1-C | reference without a model | OCTDL MH | 39 | 17 | 1 | 0 | 38 |  | share_called_any_rare (R1) 0.974 |  |

## 8. MIRAGE minus Inception-v3 K mean, each development query left out (E1h)

| Set | Full | Smallest | Image | Largest | Image |
|---|---|---|---|---|---|
| locked | 0.116 | 0.109 | data/raw/rare/MH/23.PNG | 0.122 | data/raw/rare/Stargardt/19.PNG |
| E1-M | 0.063 | 0.058 | data/raw/rare/MacTel/29.PNG | 0.068 | data/raw/rare/Stargardt/10.PNG |

Not allowed: that size or framing "explains" or "does not explain" the locked advantage (say "survives" or "does not survive this control"); "keeps" without the mean beside it; any reading of K = 16; E1-L or E1-C as the main result; "robust to framing" beyond these three variants.
